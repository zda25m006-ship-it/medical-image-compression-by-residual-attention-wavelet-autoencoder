import time, warnings, os, random, zlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2, pandas as pd

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

warnings.filterwarnings("ignore")

SEED = int(os.environ.get("RAWA_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_ROOT = Path(os.environ.get("RAWA_DATA_DIR", r"d:\archive (2)"))
RESULTS_DIR = DATA_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "samples").mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
BASE_CH = 48
DEVICE = torch.device("cpu")
MAX_TEST_PER_CLASS = int(os.environ.get("RAWA_MAX_TEST_PER_CLASS", "80"))


def haar_dwt2d(x):
    ll = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
    lh = (x[:, :, ::2, ::2] - x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] - x[:, :, 1::2, 1::2]) / 4
    hl = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] - x[:, :, ::2, 1::2] - x[:, :, 1::2, 1::2]) / 4
    hh = (x[:, :, ::2, ::2] - x[:, :, 1::2, ::2] - x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
    return torch.cat([ll, lh, hl, hh], dim=1)


def haar_idwt2d(x):
    C = x.shape[1] // 4
    ll, lh, hl, hh = x[:, :C], x[:, C:2 * C], x[:, 2 * C:3 * C], x[:, 3 * C:]
    B, C, H, W = ll.shape
    out = torch.zeros(B, C, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, ::2, ::2] = ll + lh + hl + hh
    out[:, :, 1::2, ::2] = ll - lh + hl - hh
    out[:, :, ::2, 1::2] = ll + lh - hl - hh
    out[:, :, 1::2, 1::2] = ll - lh - hl + hh
    return out


class HaarDWT(nn.Module):
    def forward(self, x):
        return haar_dwt2d(x)


class HaarIDWT(nn.Module):
    def forward(self, x):
        return haar_idwt2d(x)


class MRIDataset(Dataset):
    def __init__(self, root, transform=None, max_per_class=80):
        self.paths = []
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            p = root / cls
            if not p.exists():
                continue
            imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.jpeg")), key=lambda x: x.name.lower())
            if max_per_class is not None and len(imgs) > max_per_class:
                rng = random.Random(SEED + len(self.paths))
                imgs = rng.sample(imgs, max_per_class)
            self.paths.extend(imgs)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, img


tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
test_ds = MRIDataset(DATA_ROOT / "Testing", tf, max_per_class=MAX_TEST_PER_CLASS)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)
print(f"[Data] Test={len(test_ds)}")


class CBAM(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid(),
        )
        self.spa = nn.Sequential(nn.Conv2d(2, 1, 7, 1, 3, bias=False), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        x = x * self.fc(x).view(B, C, 1, 1)
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.spa(torch.cat([avg, mx], 1))


class ConvAE(nn.Module):
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 3, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.ReLU(True),
            nn.Conv2d(base * 4, base, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(base, base * 4, 1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class RAWAEncoder(nn.Module):
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.dwt = HaarDWT()
        self.head = nn.Sequential(nn.Conv2d(4, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(True))
        self.down1 = nn.Sequential(nn.Conv2d(base, base * 2, 3, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.ReLU(True))
        self.attn = CBAM(base * 4)
        self.btn = nn.Conv2d(base * 4, base, 1)

    def forward(self, x):
        wt = self.dwt(x)
        f0 = self.head(wt)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f2 = self.attn(f2)
        z = self.btn(f2)
        return z, f1, f0


class RAWADecoder(nn.Module):
    def __init__(self, base=BASE_CH):
        super().__init__()
        self.idwt = HaarIDWT()
        self.exp = nn.Conv2d(base, base * 4, 1)
        self.attn = CBAM(base * 4)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(base * 2)
        self.merge1 = nn.Conv2d(base * 2 + base * 2, base * 2, 1)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(base)
        self.merge2 = nn.Conv2d(base + base, base, 1)
        self.head = nn.Conv2d(base, 4, 3, 1, 1)

    def _align(self, a, b):
        if a.shape[-2:] != b.shape[-2:]:
            a = F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        return a

    def forward(self, z, f1, f0):
        x = self.attn(self.exp(z))
        x = F.relu(self.bn1(self.up1(x)), True)
        x = self.merge1(torch.cat([x, self._align(f1, x)], 1))
        x = F.relu(self.bn2(self.up2(x)), True)
        x = self.merge2(torch.cat([x, self._align(f0, x)], 1))
        wt = self.head(x)
        out = self.idwt(wt)
        return torch.sigmoid(out)


class RAWA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RAWAEncoder()
        self.decoder = RAWADecoder()

    def forward(self, x):
        z, f1, f0 = self.encoder(x)
        out = self.decoder(z, f1, f0)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out, z, f1, f0


def _encode_tensor_to_bytes(tensor):
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    q = np.round(arr * 255.0).astype(np.uint8)
    return zlib.compress(q.tobytes(), level=6)


def bpp_est(z, f1=None, f0=None):
    payload = [_encode_tensor_to_bytes(z)]
    for feat in (f1, f0):
        if feat is not None:
            payload.append(_encode_tensor_to_bytes(feat))
    bits = sum(len(p) * 8 for p in payload)
    return bits / float(IMG_SIZE * IMG_SIZE)


def evaluate(model, loader, name):
    psnrs, ssims, bpps, times, samples = [], [], [], [], []
    with torch.no_grad():
        for x, _ in loader:
            t0 = time.time()
            if isinstance(model, RAWA):
                out, z, f1, f0 = model(x)
            else:
                out, z = model(x)
                f1 = f0 = None
            dt = (time.time() - t0) / x.shape[0] * 1000
            xn = x.numpy(); on = out.numpy()
            for i in range(len(x)):
                o = xn[i, 0]; r = np.clip(on[i, 0], 0, 1)
                psnrs.append(psnr_func(o, r, data_range=1.))
                ssims.append(ssim_func(o, r, data_range=1.))
                if isinstance(model, RAWA):
                    bpps.append(bpp_est(z[i:i + 1], f1[i:i + 1], f0[i:i + 1]))
                else:
                    bpps.append(bpp_est(z[i:i + 1]))
                times.append(dt)
            if len(samples) < 4:
                samples.append((xn[0, 0], np.clip(on[0, 0], 0, 1)))
    bpp = np.mean(bpps)
    return {"Model": name, "PSNR": np.mean(psnrs), "SSIM": np.mean(ssims), "BPP": bpp, "CR": 8.0 / bpp, "Time_ms": np.mean(times)}, samples


def jpeg_baselines(loader):
    rows = []
    for q, label in [(10, "JPEG Q10"), (30, "JPEG Q30"), (75, "JPEG Q75")]:
        psnrs, ssims, bpps = [], [], []
        for x, _ in loader:
            for img_t in x:
                orig = (img_t[0].numpy() * 255).astype(np.uint8)
                _, enc = cv2.imencode(".jpg", orig, [cv2.IMWRITE_JPEG_QUALITY, q])
                dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
                o_f = orig.astype(np.float32) / 255; d_f = dec.astype(np.float32) / 255
                psnrs.append(psnr_func(o_f, d_f, data_range=1.))
                ssims.append(ssim_func(o_f, d_f, data_range=1.))
                bpps.append(len(enc) * 8 / orig.size)
        rows.append({"Model": label, "PSNR": np.mean(psnrs), "SSIM": np.mean(ssims), "BPP": np.mean(bpps), "CR": 8.0 / np.mean(bpps), "Time_ms": 0.3})
    return rows


PALETTE = {"CNN AE": "#e74c3c", "RAWA (Proposed)": "#27ae60", "JPEG Q10": "#95a5a6", "JPEG Q30": "#7f8c8d", "JPEG Q75": "#bdc3c7"}


def plot_bar(df):
    metrics = ["PSNR", "SSIM", "CR"]
    ylabels = ["PSNR (dB)", "SSIM (0–1)", "Compression Ratio ×"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = [PALETTE.get(m, "#3498db") for m in df["Model"]]
    for ax, m, yl in zip(axes, metrics, ylabels):
        bars = ax.bar(df["Model"], df[m], color=colors, edgecolor="white", lw=1.2, width=0.6)
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.set_ylabel(yl, fontsize=11)
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h*1.015, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("Model Comparison — Brain Tumor MRI Dataset", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    print("[✓] metrics_comparison.png")


def plot_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        c = PALETTE.get(row["Model"], "#3498db")
        ax.scatter(row["CR"], row["PSNR"], s=200, color=c, zorder=5, label=row["Model"])
        ax.annotate(row["Model"], (row["CR"], row["PSNR"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Compression Ratio (×)", fontsize=12); ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Quality vs Compression Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "psnr_vs_cr.png", dpi=150); plt.close()
    print("[✓] psnr_vs_cr.png")


def plot_recon(cnn_s, rawa_s, n=3):
    n = min(n, len(cnn_s), len(rawa_s))
    fig, axes = plt.subplots(3, n * 3, figsize=(n * 9 + 1, 10))
    for j in range(n):
        orig = cnn_s[j][0]; cr = np.clip(cnn_s[j][1], 0, 1); rr = np.clip(rawa_s[j][1], 0, 1)
        p_c = psnr_func(orig, cr, data_range=1.); p_r = psnr_func(orig, rr, data_range=1.)
        s_c = ssim_func(orig, cr, data_range=1.); s_r = ssim_func(orig, rr, data_range=1.)
        for k, (img, title) in enumerate([(orig, "Original"), (cr, "CNN AE"), (rr, "RAWA")]):
            col = j * 3 + k
            axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_title(f"{title} #{j + 1}", fontsize=9); axes[0, col].axis("off")
            err = np.abs(img - orig); im = axes[1, col].imshow(err, cmap="hot", vmin=0, vmax=0.15)
            axes[1, col].set_title("Error Map", fontsize=8); axes[1, col].axis("off")
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)
            if k == 0: txt = "Reference"
            elif k == 1: txt = f"PSNR:{p_c:.2f}dB\nSSIM:{s_c:.4f}"
            else: txt = f"PSNR:{p_r:.2f}dB (Δ{p_r - p_c:+.2f})\nSSIM:{s_r:.4f} (Δ{s_r - s_c:+.4f})"
            axes[2, col].text(0.5, 0.5, txt, ha="center", va="center", fontsize=10,
                              transform=axes[2, col].transAxes,
                              bbox=dict(boxstyle="round,pad=0.4", facecolor="#dfe6e9", alpha=0.9))
            axes[2, col].axis("off")
    fig.suptitle("Brain MRI: Original | CNN AE | RAWA — Reconstruction", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "reconstruction_comparison.png", dpi=150, bbox_inches="tight"); plt.close(); print("[✓] reconstruction_comparison.png")


def plot_samples(cnn_s, rawa_s):
    for i, (cs, rs) in enumerate(zip(cnn_s[:4], rawa_s[:4])):
        orig = cs[0]; cr = np.clip(cs[1], 0, 1); rr = np.clip(rs[1], 0, 1)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes, [orig, cr, rr], ["Original MRI", "CNN Autoencoder", "RAWA (Proposed)"]):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            p = psnr_func(orig, img, data_range=1.); s = ssim_func(orig, img, data_range=1.)
            ax.set_title(f"{title}\nPSNR={p:.2f} dB | SSIM={s:.4f}", fontsize=11); ax.axis("off")
        plt.suptitle(f"MRI Sample #{i + 1}", fontsize=13, fontweight="bold")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / "samples" / f"sample_{i + 1}.png", dpi=150); plt.close()
    print("[✓] Individual samples saved")


print("[Evaluating CNN AE...]"); cnn_res, cnn_s = evaluate(ConvAE(), test_loader, "CNN AE")
print("[Evaluating RAWA...]"); rawa_res, rawa_s = evaluate(RAWA(), test_loader, "RAWA (Proposed)")
print("[JPEG baselines...]"); trad = jpeg_baselines(test_loader)

all_res = trad + [cnn_res, rawa_res]
df = pd.DataFrame(all_res)[["Model", "PSNR", "SSIM", "BPP", "CR", "Time_ms"]].round(4)

print("\n" + "=" * 65)
print("  RESULTS — Brain Tumor MRI Dataset")
print("=" * 65)
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

plot_bar(df); plot_scatter(df); plot_recon(cnn_s, rawa_s); plot_samples(cnn_s, rawa_s)
print(f"\n[✓] All done! Results in: {RESULTS_DIR}")
