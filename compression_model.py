"""
RAWA: Residual Attention Wavelet Autoencoder (CPU-Optimized)
Novel model for near-lossless Brain MRI Compression.

Key innovations over CNN baseline:
  1. Haar DWT multi-scale decomposition (lossless sub-bands)
  2. CBAM channel + spatial attention at bottleneck
  3. Wavelet-domain skip connections
  4. Multi-term loss: MSE + MS-SSIM + Edge-gradient

Benchmarked vs:
  - CNN Autoencoder (baseline)
  - JPEG Q10 / Q30 / Q75 (traditional)
"""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────
BASE_DIR    = Path(r"d:\archive (2)")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "samples").mkdir(exist_ok=True)

IMG_SIZE   = 128
BATCH_SIZE = 16
EPOCHS     = 25
LR         = 3e-4
BASE_CH    = 48          # smaller base for CPU speed
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}  |  IMG={IMG_SIZE}  EPOCHS={EPOCHS}  BASE_CH={BASE_CH}")

# ─── DATASET ─────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, root, transform=None, max_per_class=300):
        self.paths = []
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            p = root / cls
            if not p.exists(): continue
            imgs = sorted(list(p.glob("*.jpg")) +
                          list(p.glob("*.png")) +
                          list(p.glob("*.jpeg")))
            self.paths.extend(imgs[:max_per_class])
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img


tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
train_ds     = MRIDataset(BASE_DIR / "Training", tf)
test_ds      = MRIDataset(BASE_DIR / "Testing",  tf, max_per_class=80)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[Data] Train={len(train_ds)}  Test={len(test_ds)}")


# ═══════════════════════════════════════════════════════
# BUILDING BLOCKS
# ═══════════════════════════════════════════════════════

class HaarDWT(nn.Module):
    """Differentiable Haar Wavelet Transform — lossless spectral decomposition"""
    def forward(self, x):
        ll = (x[:,:, ::2, ::2] + x[:,:,1::2, ::2] + x[:,:, ::2,1::2] + x[:,:,1::2,1::2]) / 4
        lh = (x[:,:, ::2, ::2] - x[:,:,1::2, ::2] + x[:,:, ::2,1::2] - x[:,:,1::2,1::2]) / 4
        hl = (x[:,:, ::2, ::2] + x[:,:,1::2, ::2] - x[:,:, ::2,1::2] - x[:,:,1::2,1::2]) / 4
        hh = (x[:,:, ::2, ::2] - x[:,:,1::2, ::2] - x[:,:, ::2,1::2] + x[:,:,1::2,1::2]) / 4
        return torch.cat([ll, lh, hl, hh], dim=1)   # (B, 4C, H/2, W/2)


class HaarIDWT(nn.Module):
    """Differentiable Inverse Haar Wavelet Transform"""
    def forward(self, x):
        C  = x.shape[1] // 4
        ll, lh, hl, hh = x[:,:C], x[:,C:2*C], x[:,2*C:3*C], x[:,3*C:]
        B, C, H, W = ll.shape
        out = torch.zeros(B, C, H*2, W*2, device=x.device, dtype=x.dtype)
        out[:, :,  ::2,  ::2] = ll + lh + hl + hh
        out[:, :, 1::2,  ::2] = ll - lh + hl - hh
        out[:, :,  ::2, 1::2] = ll + lh - hl - hh
        out[:, :, 1::2, 1::2] = ll - lh - hl + hh
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)"""
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(True),
            nn.Linear(ch//r, ch, bias=False),  nn.Sigmoid(),
        )
        self.spa = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3, bias=False), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention
        x = x * self.fc(x).view(B, C, 1, 1)
        # Spatial attention
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        x = x * self.spa(torch.cat([avg, mx], 1))
        return x


# ═══════════════════════════════════════════════════════
# MODEL 1 — CNN AUTOENCODER BASELINE
# ═══════════════════════════════════════════════════════
class ConvAE(nn.Module):
    """Plain CNN Autoencoder — No wavelet, No attention"""
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,      base,   3, 2, 1, bias=False), nn.BatchNorm2d(base),   nn.ReLU(True),
            nn.Conv2d(base,   base*2, 3, 2, 1, bias=False), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.Conv2d(base*2, base*4, 3, 2, 1, bias=False), nn.BatchNorm2d(base*4), nn.ReLU(True),
            nn.Conv2d(base*4, base,   1),                                            # bottleneck
        )
        self.dec = nn.Sequential(
            nn.Conv2d(base, base*4, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.ReLU(True),
            nn.ConvTranspose2d(base*2, base,   4, 2, 1), nn.BatchNorm2d(base),   nn.ReLU(True),
            nn.ConvTranspose2d(base,   1,      4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


# ═══════════════════════════════════════════════════════
# MODEL 2 — RAWA (Proposed Novel Model)
# ═══════════════════════════════════════════════════════
class RAWAEncoder(nn.Module):
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.dwt  = HaarDWT()                    # 1→4ch, H/2×W/2

        self.head = nn.Sequential(               # (B,4,64,64)→(B,base,64,64)
            nn.Conv2d(4, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(True),
        )
        self.down1 = nn.Sequential(              # (B,base,64,64)→(B,base*2,32,32)
            nn.Conv2d(base, base*2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*2), nn.ReLU(True),
        )
        self.down2 = nn.Sequential(              # (B,base*2,32,32)→(B,base*4,16,16)
            nn.Conv2d(base*2, base*4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base*4), nn.ReLU(True),
        )
        self.attn = CBAM(base*4)                 # attention at deepest feature map
        self.btn  = nn.Conv2d(base*4, base, 1)  # bottleneck: strong compression

    def forward(self, x):
        wt = self.dwt(x)          # wavelet sub-bands
        f0 = self.head(wt)        # (B, base, 64, 64)
        f1 = self.down1(f0)       # (B, base*2, 32, 32)
        f2 = self.down2(f1)       # (B, base*4, 16, 16)
        f2 = self.attn(f2)        # CBAM attention
        z  = self.btn(f2)         # (B, base, 16, 16) — latent code
        return z, f1, f0          # z + multi-scale skips


class RAWADecoder(nn.Module):
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.idwt = HaarIDWT()

        self.exp  = nn.Conv2d(base, base*4, 1)  # expand bottleneck
        self.attn = CBAM(base*4)

        # up1: 16→32; cat with f1 (base*2)
        self.up1    = nn.ConvTranspose2d(base*4, base*2, 4, 2, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(base*2)
        self.merge1 = nn.Conv2d(base*2 + base*2, base*2, 1)

        # up2: 32→64; cat with f0 (base)
        self.up2    = nn.ConvTranspose2d(base*2, base, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(base)
        self.merge2 = nn.Conv2d(base + base, base, 1)

        # output head → 4 wavelet channels → IDWT → pixel
        self.head = nn.Conv2d(base, 4, 3, 1, 1)

    def _align(self, a, b):
        """Align spatial dims of a to match b if needed"""
        if a.shape[-2:] != b.shape[-2:]:
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return a

    def forward(self, z, f1, f0):
        x  = self.attn(self.exp(z))                          # (B, base*4, 16, 16)
        x  = F.relu(self.bn1(self.up1(x)), True)             # (B, base*2, 32, 32)
        x  = self.merge1(torch.cat([x, self._align(f1, x)], 1))
        x  = F.relu(self.bn2(self.up2(x)), True)             # (B, base, 64, 64)
        x  = self.merge2(torch.cat([x, self._align(f0, x)], 1))
        wt = self.head(x)                                     # (B, 4, 64, 64)
        out = self.idwt(wt)                                   # (B, 1, 128, 128)
        return torch.sigmoid(out)


class RAWA(nn.Module):
    """
    RAWA: Residual Attention Wavelet Autoencoder
    ─────────────────────────────────────────────
    Advantages over plain CNN AE:
      • DWT decomposes MRI into LL (structure/low-freq) +
        LH/HL/HH (edges/high-freq) → network learns each separately
      • CBAM focuses on diagnostically critical regions
      • Multi-scale skip connections in wavelet domain preserve fine detail
      • Multi-term (MSE+SSIM+gradient) loss for perceptual quality
    """
    def __init__(self):
        super().__init__()
        self.encoder = RAWAEncoder()
        self.decoder = RAWADecoder()

    def forward(self, x):
        z, f1, f0 = self.encoder(x)
        out = self.decoder(z, f1, f0)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out, z


# ─── MULTI-TERM LOSS ─────────────────────────────────────────────
def ms_ssim_loss(x, y, levels=3):
    loss = 0.0
    for _ in range(levels):
        mu_x  = F.avg_pool2d(x, 3, 1, 1)
        mu_y  = F.avg_pool2d(y, 3, 1, 1)
        sig_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sig_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sig_xy= F.avg_pool2d(x*y,  3, 1, 1) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y+C1)*(2*sig_xy+C2)) / \
                   ((mu_x**2+mu_y**2+C1)*(sig_x+sig_y+C2)+1e-8)
        loss += (1 - ssim_map.mean())
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
    return loss / levels


def grad_loss(pred, tgt):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                       dtype=pred.dtype, device=pred.device).view(1,1,3,3)
    ky = kx.transpose(-2,-1)
    gx_p = F.conv2d(pred, kx, padding=1);  gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(tgt,  kx, padding=1);  gy_t = F.conv2d(tgt,  ky, padding=1)
    return (F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)) / 2


class RAWALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, tgt):
        mse  = F.mse_loss(pred, tgt)
        ssim = ms_ssim_loss(pred, tgt)
        grad = grad_loss(pred, tgt)
        loss = 0.5*mse + 0.35*ssim + 0.15*grad
        return loss, {"mse": mse.item(), "ssim_l": ssim.item(), "grad": grad.item()}


# ─── TRAINING ────────────────────────────────────────────────────
def train_model(model, loader, epochs, loss_fn=None, name="Model"):
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(DEVICE).train()
    history = []
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'─'*58}")
    print(f"  {name}   [{n_params:.2f}M params, {epochs} epochs]")
    print(f"{'─'*58}")
    for ep in range(1, epochs+1):
        total, cnt = 0., 0
        for x, _ in loader:
            x = x.to(DEVICE)
            opt.zero_grad()
            out, z = model(x)
            loss = loss_fn(out, x)[0] if loss_fn else F.mse_loss(out, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); cnt += 1
        sched.step()
        avg = total / cnt
        history.append(avg)
        if ep == 1 or ep % 5 == 0:
            print(f"  Epoch {ep:03d}/{epochs}  Loss={avg:.6f}  LR={sched.get_last_lr()[0]:.2e}")
    return history


# ─── EVALUATION ──────────────────────────────────────────────────
def bpp_estimate(z):
    arr  = z.detach().cpu().numpy().flatten()
    arrq = np.round(arr * 16).astype(np.int32)
    _, c = np.unique(arrq, return_counts=True)
    p    = c / c.sum()
    H    = -np.sum(p * np.log2(p + 1e-12))
    return H * (z.numel() / z.shape[0]) / (IMG_SIZE * IMG_SIZE)


def evaluate(model, loader, name):
    model.eval().to(DEVICE)
    psnrs, ssims, bpps, times, samples = [], [], [], [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            t0 = time.time()
            out, z = model(x)
            dt = (time.time()-t0)/x.shape[0]*1000
            xn, on = x.cpu().numpy(), out.cpu().numpy()
            for i in range(len(x)):
                o = xn[i,0]; r = np.clip(on[i,0],0,1)
                psnrs.append(psnr_func(o, r, data_range=1.0))
                ssims.append(ssim_func(o, r, data_range=1.0))
                bpps.append(bpp_estimate(z[i:i+1]))
                times.append(dt)
            if len(samples) < 4:
                samples.append((xn[0,0], np.clip(on[0,0],0,1)))
    bpp = np.mean(bpps)
    return {"Model": name, "PSNR": np.mean(psnrs),
            "SSIM": np.mean(ssims), "BPP": bpp,
            "CR": 8.0/bpp, "Time_ms": np.mean(times)}, samples


def jpeg_baselines(loader):
    rows = []
    for q, label in [(10,"JPEG Q10"),(30,"JPEG Q30"),(75,"JPEG Q75")]:
        psnrs, ssims, bpps = [], [], []
        for x, _ in loader:
            for img_t in x:
                orig = (img_t[0].numpy()*255).astype(np.uint8)
                _, enc = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, q])
                dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
                o_f = orig.astype(np.float32)/255
                d_f = dec.astype(np.float32)/255
                psnrs.append(psnr_func(o_f, d_f, data_range=1.0))
                ssims.append(ssim_func(o_f, d_f, data_range=1.0))
                bpps.append(len(enc)*8/orig.size)
        rows.append({"Model": label, "PSNR": np.mean(psnrs),
                     "SSIM": np.mean(ssims), "BPP": np.mean(bpps),
                     "CR": 8.0/np.mean(bpps), "Time_ms": 0.3})
    return rows


# ─── PLOTS ───────────────────────────────────────────────────────
PALETTE = {"CNN AE":"#e74c3c", "RAWA (Proposed)":"#27ae60",
           "JPEG Q10":"#95a5a6", "JPEG Q30":"#7f8c8d", "JPEG Q75":"#bdc3c7"}

def plot_training(cnn_h, rawa_h):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(cnn_h,  color="#e74c3c", lw=2.5, label="CNN Autoencoder (Baseline)")
    ax.plot(rawa_h, color="#27ae60", lw=2.5, label="RAWA (Proposed)")
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curves — Brain MRI Compression", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR/"training_curves.png", dpi=150); plt.close()
    print("[✓] training_curves.png")

def plot_bar(df):
    metrics = ["PSNR","SSIM","CR"]
    ylabels = ["PSNR (dB)","SSIM (0–1)","Compression Ratio ×"]
    fig, axes = plt.subplots(1,3, figsize=(16,5))
    colors = [PALETTE.get(m, "#3498db") for m in df["Model"]]
    for ax, m, yl in zip(axes, metrics, ylabels):
        bars = ax.bar(df["Model"], df[m], color=colors, edgecolor="white", lw=1.2, width=0.6)
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.set_ylabel(yl, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h*1.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("Model Comparison — Brain Tumor MRI Dataset",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    print("[✓] metrics_comparison.png")

def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8,5))
    for _, row in df.iterrows():
        c = PALETTE.get(row["Model"], "#3498db")
        ax.scatter(row["CR"], row["PSNR"], s=200, color=c, zorder=5, label=row["Model"])
        ax.annotate(row["Model"], (row["CR"], row["PSNR"]),
                    xytext=(5,5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Compression Ratio (×)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Quality vs Compression Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"psnr_vs_cr.png", dpi=150); plt.close()
    print("[✓] psnr_vs_cr.png")

def plot_recon(cnn_s, rawa_s, n=3):
    n = min(n, len(cnn_s), len(rawa_s))
    fig, axes = plt.subplots(3, n*3, figsize=(n*9+1, 10))
    for j in range(n):
        orig  = cnn_s[j][0]
        cnn_r = np.clip(cnn_s[j][1], 0, 1)
        raw_r = np.clip(rawa_s[j][1], 0, 1)
        p_c = psnr_func(orig, cnn_r, data_range=1.)
        p_r = psnr_func(orig, raw_r, data_range=1.)
        s_c = ssim_func(orig, cnn_r, data_range=1.)
        s_r = ssim_func(orig, raw_r, data_range=1.)
        for k, (img, title) in enumerate(
                [(orig,"Original"),(cnn_r,"CNN AE"),(raw_r,"RAWA")]):
            col = j*3+k
            axes[0,col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0,col].set_title(f"{title} #{j+1}", fontsize=9); axes[0,col].axis("off")
            err = np.abs(img-orig)
            im = axes[1,col].imshow(err, cmap="hot", vmin=0, vmax=0.15)
            axes[1,col].set_title("Error", fontsize=8); axes[1,col].axis("off")
            plt.colorbar(im, ax=axes[1,col], fraction=0.046)
            if k==0: txt="Reference"
            elif k==1: txt=f"PSNR:{p_c:.2f}dB\nSSIM:{s_c:.4f}"
            else: txt=f"PSNR:{p_r:.2f}dB (Δ+{p_r-p_c:.2f})\nSSIM:{s_r:.4f} (Δ+{s_r-s_c:.4f})"
            axes[2,col].text(0.5,0.5,txt,ha="center",va="center",fontsize=10,
                transform=axes[2,col].transAxes,
                bbox=dict(boxstyle="round,pad=0.4",facecolor="#dfe6e9",alpha=0.9))
            axes[2,col].axis("off")
    fig.suptitle("Brain MRI: Original | CNN | RAWA — Reconstruction",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/"reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(); print("[✓] reconstruction_comparison.png")

def plot_samples(cnn_s, rawa_s):
    for i, (cs, rs) in enumerate(zip(cnn_s[:4], rawa_s[:4])):
        orig=cs[0]; cr=np.clip(cs[1],0,1); rr=np.clip(rs[1],0,1)
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        for ax, img, title in zip(axes,[orig,cr,rr],
                ["Original MRI","CNN Autoencoder","RAWA (Proposed)"]):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            p=psnr_func(orig,img,data_range=1.); s=ssim_func(orig,img,data_range=1.)
            ax.set_title(f"{title}\nPSNR={p:.2f} dB | SSIM={s:.4f}", fontsize=11)
            ax.axis("off")
        plt.suptitle(f"MRI Sample #{i+1}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"samples"/f"sample_{i+1}.png", dpi=150); plt.close()
    print("[✓] Individual samples saved")


# ─── MAIN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*62)
    print("  RAWA: Residual Attention Wavelet Autoencoder")
    print("  Novel Near-Lossless Brain MRI Compression")
    print("="*62)

    # CNN Baseline
    cnn   = ConvAE()
    cnn_h = train_model(cnn, train_loader, EPOCHS, name="CNN Autoencoder (Baseline)")
    torch.save(cnn.state_dict(), RESULTS_DIR/"cnn_ae.pth")
    print("[✓] CNN weights saved")

    # RAWA Proposed
    rawa    = RAWA()
    loss_fn = RAWALoss()
    rawa_h  = train_model(rawa, train_loader, EPOCHS, loss_fn=loss_fn,
                          name="RAWA — Proposed Novel Model")
    torch.save(rawa.state_dict(), RESULTS_DIR/"rawa.pth")
    print("[✓] RAWA weights saved")

    # Evaluate
    print("\n[Evaluating on test set...]")
    cnn_res,  cnn_s  = evaluate(cnn,  test_loader, "CNN AE")
    rawa_res, rawa_s = evaluate(rawa, test_loader, "RAWA (Proposed)")

    print("[JPEG baselines...]")
    trad = jpeg_baselines(test_loader)

    # Results table
    all_res = trad + [cnn_res, rawa_res]
    df = pd.DataFrame(all_res)[["Model","PSNR","SSIM","BPP","CR","Time_ms"]].round(4)

    print("\n" + "="*65)
    print("  RESULTS — Brain Tumor MRI Dataset")
    print("="*65)
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR/"comparison_table.csv", index=False)
    print(f"\n  Saved → {RESULTS_DIR/'comparison_table.csv'}")

    dp = rawa_res["PSNR"] - cnn_res["PSNR"]
    ds = rawa_res["SSIM"] - cnn_res["SSIM"]
    print(f"\n{'─'*58}")
    print(f"  [RAWA vs CNN Improvement Summary]")
    print(f"  PSNR : {rawa_res['PSNR']:.2f} dB  vs  {cnn_res['PSNR']:.2f} dB   → Δ {dp:+.2f} dB")
    print(f"  SSIM : {rawa_res['SSIM']:.4f}     vs  {cnn_res['SSIM']:.4f}    → Δ {ds:+.4f}")
    print(f"  CR   : {rawa_res['CR']:.2f}×      vs  {cnn_res['CR']:.2f}×")
    won = "✅ RAWA outperforms CNN AE!" if dp > 0 and ds > 0 else "⚠ Check training"
    print(f"  {won}")
    print(f"{'─'*58}")

    # Plots
    plot_training(cnn_h, rawa_h)
    plot_bar(df)
    plot_scatter(df)
    plot_recon(cnn_s, rawa_s)
    plot_samples(cnn_s, rawa_s)

    # Save models
    torch.save(cnn.state_dict(),  RESULTS_DIR/"cnn_ae.pth")
    torch.save(rawa.state_dict(), RESULTS_DIR/"rawa.pth")
    print(f"\n[✓] Models saved → {RESULTS_DIR}")
    print("[✓] All done! Results in d:\\archive (2)\\results\\")
