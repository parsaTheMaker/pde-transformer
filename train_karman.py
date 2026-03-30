"""
train_karman.py
===============
Train PDE-Transformer (MC-S) on the Kármán vortex CFD data in data/sim_000000/.

Data layout expected
--------------------
  velocity_XXXXXX.npz  →  arr_0 shape (2, 256, 128)   [vx, vy]
  pressure_XXXXXX.npz  →  arr_0 shape (1, 256, 128)   [p]

Task
----
  Given frame t → predict frame t+1
  Input:  (vx, vy, p)  at  t      →  3 channels
  Target: (vx, vy, p)  at  t+1

Training
--------
  30 epochs, AdamW, MSE loss
  80/20 train/val split (chronological, NOT random, to avoid data leakage)
  Batch size 4 with gradient accumulation × 4 → effective batch 16

Outputs (all in ./runs/karman/)
--------------------------------
  loss_curve.png        – train/val MSE per epoch
  pred_vs_gt.mp4        – side-by-side autoregressive rollout vs ground truth at final epoch
  best_model.pt         – best validation checkpoint
"""

import os, glob, sys, math

# ─── Fix: Resolve CUDNN_STATUS_NOT_INITIALIZED while keeping cuDNN enabled
# Must be set before 'import torch' to ensure binary linkage is correct.
os.environ["LD_LIBRARY_PATH"] = (
    "/home/vatani/miniconda/envs/ag_env/lib/python3.11/site-packages/nvidia/cudnn/lib:" + 
    os.environ.get("LD_LIBRARY_PATH", "")
)

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
try:
    import torch.nn.attention.flex_attention
    if not hasattr(torch.nn.attention.flex_attention, "AuxRequest"):
        class AuxRequest: pass
        torch.nn.attention.flex_attention.AuxRequest = AuxRequest
except (ImportError, AttributeError):
    pass

# ─── Fix: Resolve ImportError for find_pruneable_heads_and_indices in transformers.pytorch_utils
# This is missing in transformers v5.x but requested by pdetransformer/core/separate_channels/pde_transformer.py
import transformers
if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(*args, **kwargs):
        return [], []
    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from tqdm import tqdm
import subprocess
from io import BytesIO
from multiprocessing import Pool, cpu_count

# ── reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────── config ──────
SIM_DIR    = "data/sim_000000"
OUT_DIR    = "runs/karman"
EPOCHS     = 10
BATCH_SIZE = 12
ACCUM_GRAD = 2           # effective batch = 8 * 8 = 64
LR         = 2e-5        # documented base learning rate
VAL_FRAC   = 0.10        # last 20 % of time-series for validation (no leakage)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CROP       = None        # No longer cropping; 256x128 is fine
FPS_VID    = 10
VID_FRAMES = 50          # autoregressive steps in prediction video
DPI_VID    = 110
SKIP_TRAIN = False       # Run full training with the fixed parameters
OS_MASK_PATH = os.path.join(SIM_DIR, "obstacle_mask.npz")

# ──────────────────────────────────────────── spatial crop helper ────────────
def centre_crop(x, size=CROP):
    """Centre-crop along the x (width=256) axis → (B, C, size, H)."""
    W = x.shape[2]
    start = (W - size) // 2
    return x[:, :, start:start+size, :]

def np_crop(x, size=CROP):
    """Centre-crop along the second-to-last axis → (..., size, H)."""
    W = x.shape[-2]
    start = (W - size) // 2
    return x[..., start:start+size, :]

# ── Load Obstacle Mask ───────────────────────────────────────────────────────
# 1 = fluid, 0 = cylinder (provided in simulations)
mask_raw = np.load(OS_MASK_PATH)["arr_0"].astype(np.float32)
# The mask is (256, 128); we no longer crop it.
MASK_NP  = mask_raw
MASK     = torch.from_numpy(MASK_NP).to(DEVICE).view(1, 1, 256, 128)

# Pre-calculate normalized representation of "physical zero" for vx, vy, p
# so we can mask effectively in normalized space.
# MEAN and STD will be computed later after loading data, so we'll init ZERO_NORM then.
ZERO_NORM = None

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Output: {OUT_DIR}/")

# ─────────────────────────────────────────────────── load & normalise data ───
print("\nLoading data …")
vel_files = sorted(glob.glob(os.path.join(SIM_DIR, "velocity_*.npz")))
pre_files = sorted(glob.glob(os.path.join(SIM_DIR, "pressure_*.npz")))
N = len(vel_files)
assert N == len(pre_files) == 1300, f"Expected 1300 files, got {N}"

# Stack into contiguous array  shape (N, 3, 256, 128)
# Use float32 & load lazily via the Dataset
# Compute global mean/std from 200 sampled frames for normalisation
print("Computing normalization statistics …")
sample_idx = np.linspace(0, N-1, 200, dtype=int)
samples = []
for i in sample_idx:
    v = np.load(vel_files[i])["arr_0"].astype(np.float32)   # (2,256,128)
    p = np.load(pre_files[i])["arr_0"].astype(np.float32)   # (1,256,128)
    samples.append(np.concatenate([v, p], axis=0))
samples = np.stack(samples)  # (200, 3, 256, 128)

MEAN = torch.tensor(samples.mean(axis=(0, 2, 3)), dtype=torch.float32)  # (3,)
STD  = torch.tensor(samples.std (axis=(0, 2, 3)), dtype=torch.float32) + 1e-6
print(f"  per-channel mean: {MEAN.numpy().round(5)}")
print(f"  per-channel std:  {STD.numpy().round(5)}")

# ─────────────────────────────────────────────────────────────── dataset ─────
class KarmanDataset(Dataset):
    """Each sample: (frame_t normalized, frame_t+1 normalized) as tensors."""
    def __init__(self, vel_files, pre_files, mean, std):
        self.vel = vel_files
        self.pre = pre_files
        self.mean = mean[:, None, None]   # (3,1,1)  broadcast-ready
        self.std  = std [:, None, None]

    def __len__(self):
        return len(self.vel) - 1          # pairs (t, t+1)

    def _load(self, idx):
        v = torch.from_numpy(np.load(self.vel[idx])["arr_0"].astype(np.float32))
        p = torch.from_numpy(np.load(self.pre[idx])["arr_0"].astype(np.float32))
        x = torch.cat([v, p], dim=0)     # (3, 256, 128)
        return (x - self.mean) / self.std

    def __getitem__(self, idx):
        return self._load(idx), self._load(idx + 1)

# chronological split (no data leakage)
split = int(N * (1 - VAL_FRAC))
train_ds = KarmanDataset(vel_files[:split],  pre_files[:split],  MEAN, STD)
val_ds   = KarmanDataset(vel_files[split-1:],pre_files[split-1:],MEAN, STD)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
print(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

# ── compute ZERO_NORM now that we have MEAN/STD ───────────────────────────────
ZERO_NORM = ((torch.zeros(3).to(DEVICE) - MEAN.to(DEVICE)) / STD.to(DEVICE)).view(1, 3, 1, 1)

# ──────────────────────────────────────────────────── model (PDE-MC-S) ───────
print("\nBuilding PDE-Transformer (MC-S) …")
from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

# MC-S spec: 256×128 input, 3 channels, patch_size=4
# The model expects square sample_size; we'll crop/pad x to 128×128 (the y dim)
# and process two halves OR use sample_size=128 and trim x to 128×128.
# Simplest: centre-crop to 128×128 (square) for training.
CROP = 128   # spatial crop size (square)

model = PDETransformer(
    sample_size=256,          # positional argument; H=256, W=128 internally handled
    in_channels=3,
    out_channels=3,
    type="PDE-S",
    patch_size=4,
    periodic=False,           # Kármán vortex is non-periodic (inflow/outflow)
    carrier_token_active=True # Enable hierarchical attention for better global flow
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  Parameters: {n_params:.1f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ─────────────────────────────────── PDE task label (cylinder flow = 12) ─────
# from metadata_remapping.py: "navier-stokes: incompressible cylinder flow" → idx 12
TASK_LABEL = torch.tensor([12], dtype=torch.long).to(DEVICE)

def get_labels(batch_size):
    return TASK_LABEL.expand(batch_size)

# ──────────────────────────────────────────────────── training loop ──────────
train_losses, val_losses = [], []
best_val = math.inf

if not SKIP_TRAIN:

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        epoch_train_loss = 0.0
        n_train = 0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(tqdm(train_loader,
                                            desc=f"Epoch {epoch:02d}/{EPOCHS} [train]",
                                            leave=False)):
            # Full 256x128; no cropping
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            labels = get_labels(x.shape[0])

            pred = model(x, class_labels=labels).sample
            # Enforce obstacle mask: fluid areas (MASK=1) keep prediction,
            # obstacle areas (MASK=0) forced to normalized zero.
            pred = pred * MASK + ZERO_NORM * (1.0 - MASK)
            
            loss = F.mse_loss(pred, y) / ACCUM_GRAD
            loss.backward()

            if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_loss += loss.item() * ACCUM_GRAD * x.shape[0]
            n_train += x.shape[0]

        train_loss = epoch_train_loss / n_train

        # ── validate ──
        model.eval()
        epoch_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader,
                              desc=f"Epoch {epoch:02d}/{EPOCHS} [val]  ",
                              leave=False):
                # Full 256x128
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                labels = get_labels(x.shape[0])
                pred = model(x, class_labels=labels).sample
                pred = pred * MASK + ZERO_NORM * (1.0 - MASK)
                loss = F.mse_loss(pred, y)
                epoch_val_loss += loss.item() * x.shape[0]
                n_val += x.shape[0]

        val_loss = epoch_val_loss / n_val
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            best_marker = " ← best"
        else:
            best_marker = ""

        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  "
              f"val_loss={val_loss:.6f}{best_marker}")
    
    # ── loss curve (only if we just trained) ──
    print("\nSaving loss curve …")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#111")
    ax.set_facecolor("#111")
    ax.plot(range(1, EPOCHS+1), train_losses, color="#00c8ff", linewidth=2, label="Train MSE")
    ax.plot(range(1, EPOCHS+1), val_losses,   color="#ff6b35", linewidth=2, label="Val MSE")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.set_title("PDE-Transformer — Kármán Vortex Street", color="white", fontsize=13)
    ax.legend(framealpha=0.3)
    ax.tick_params(colors="white")
    ax.set_yscale("log")
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    plt.tight_layout()
    curve_path = os.path.join(OUT_DIR, "loss_curve.png")
    plt.savefig(curve_path, dpi=150, facecolor="#111")
    plt.close()
    print(f"  Saved: {curve_path}")
else:
    print("\nSkipping training (SKIP_TRAIN=True).")


# ──────────────────────────────────── prediction vs GT rollout ───────────────

print("\nGenerating prediction vs ground-truth rollout video …")

# Load best model weights for rollout
if not os.path.exists(os.path.join(OUT_DIR, "best_model.pt")):
    print("No best_model.pt found, skipping video.")
else:
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pt"), map_location=DEVICE))
model.eval()

# Ground-truth sequence from the val split: first VID_FRAMES+1 frames
gt_frames = []
for i in range(min(VID_FRAMES + 1, len(val_ds.vel))):
    v = np.load(val_ds.vel[i])["arr_0"].astype(np.float32)
    p = np.load(val_ds.pre[i])["arr_0"].astype(np.float32)
    frame = np.concatenate([v, p], axis=0)  # (3,256,128)
    gt_frames.append(frame)
gt_frames = np.stack(gt_frames)   # (T, 3, 256, 128)

# normalise
mean_np = MEAN.numpy()[:, None, None]
std_np  = STD.numpy()[:, None, None]
gt_norm = (gt_frames - mean_np) / std_np

# autoregressive prediction
pred_frames = [gt_norm[0]]
with torch.no_grad():
    current = torch.tensor(gt_norm[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    labels  = get_labels(1)
    for _ in range(VID_FRAMES):
        nxt = model(current, class_labels=labels).sample
        # Strictly enforce obstacle mask during rollout
        nxt = nxt * MASK + ZERO_NORM * (1.0 - MASK)
        pred_frames.append(nxt[0].cpu().numpy())
        current = nxt

pred_frames = np.stack(pred_frames)  # (T, 3, 256, 128)
gt_norm_seq = gt_norm[:len(pred_frames)]

# un-normalise for display (use velocity magnitude + pressure)
def unnorm(x_norm, ch):
    """x_norm: (3, H, W);  returns channel ch unnormalised."""
    return x_norm[ch] * std_np[ch, 0, 0] + mean_np[ch, 0, 0]

# ── render frames and pipe to ffmpeg ─────────────────────────────────────────
FIG_W, FIG_H = 12, 5.5
cmap_vel  = "viridis"
cmap_pres = "RdBu_r"

def vel_mag(frame_3chw):
    vx = unnorm(frame_3chw, 0)
    vy = unnorm(frame_3chw, 1)
    return np.sqrt(vx**2 + vy**2)

def pres(frame_3chw):
    return unnorm(frame_3chw, 2)

# global colour limits from GT
gt_vel_max = np.percentile([vel_mag(gt_norm_seq[i]) for i in range(len(pred_frames))], 100)
gt_pres_all = np.concatenate([pres(gt_norm_seq[i]).ravel() for i in range(len(pred_frames))])
gt_pres_abs = np.percentile(np.abs(gt_pres_all), 100)

norm_vel  = Normalize(vmin=0,           vmax=gt_vel_max)
norm_pres = Normalize(vmin=-gt_pres_abs, vmax=gt_pres_abs)

# Ensure even dimensions for ffmpeg compatibility
W_px = int(FIG_W * DPI_VID)
H_px = int(FIG_H * DPI_VID)
W_px -= W_px % 2
H_px -= H_px % 2

ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-vcodec", "rawvideo",
    "-s", f"{W_px}x{H_px}", "-pix_fmt", "rgba",
    "-r", str(FPS_VID), "-i", "pipe:0",
    "-vcodec", "libx264", "-crf", "18",
    "-preset", "fast", "-pix_fmt", "yuv420p",
    "-loglevel", "error",
    os.path.join(OUT_DIR, "pred_vs_gt.mp4"),
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

n_frames_vid = len(pred_frames)
for i in tqdm(range(n_frames_vid), desc="Rendering prediction video"):
    fig = plt.figure(figsize=(W_px/DPI_VID, H_px/DPI_VID), dpi=DPI_VID, facecolor="#111")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.05, right=0.95, top=0.88, bottom=0.08,
                           hspace=0.45, wspace=0.10)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    titles = ["GT  |u|", "Pred  |u|", "GT  p", "Pred  p"]
    data   = [
        vel_mag(gt_norm_seq[i]),
        vel_mag(pred_frames[i]),
        pres(gt_norm_seq[i]),
        pres(pred_frames[i]),
    ]
    cmaps  = [cmap_vel, cmap_vel, cmap_pres, cmap_pres]
    norms  = [norm_vel, norm_vel, norm_pres, norm_pres]

    for ax, d, t, c, n in zip(axes, data, titles, cmaps, norms):
        ax.set_facecolor("#111")
        im = ax.imshow(np.rot90(d, k=1), origin="lower", aspect="auto",
                       cmap=c, norm=n, interpolation="bilinear")
        ax.set_title(t, color="white", fontsize=9, fontweight="bold", pad=3)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.05, fraction=0.05, format="%.3f")
        cb.ax.tick_params(colors="#ccc", labelsize=6)
        cb.outline.set_edgecolor("#555")

    fig.text(0.5, 0.94,
             f"t = {i:03d}   |   Val frame {i:03d}/{n_frames_vid-1}",
             ha="center", va="top", color="white", fontsize=9,
             fontfamily="monospace")

    buf = BytesIO()
    fig.savefig(buf, format="rgba", dpi=DPI_VID)
    plt.close(fig)
    buf.seek(0)
    ffmpeg_proc.stdin.write(buf.read())

ffmpeg_proc.stdin.close()
ret = ffmpeg_proc.wait()
if ret != 0:
    print(f"ffmpeg error code {ret}")
else:
    print(f"  Saved: {OUT_DIR}/pred_vs_gt.mp4")

# ── final summary ─────────────────────────────────────────────────────────────
if not SKIP_TRAIN:
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Best val MSE : {best_val:.6f}")
    print(f"  Loss curve   : {OUT_DIR}/loss_curve.png")
    print(f"  Prediction   : {OUT_DIR}/pred_vs_gt.mp4")
    print(f"  Checkpoint   : {OUT_DIR}/best_model.pt")
    print(f"{'='*60}\n")
    
    # Print epoch table
    print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}")
    print("-" * 35)
    for ep, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
        print(f"{ep:>6}  {tl:>12.6f}  {vl:>12.6f}")
else:
    print(f"\n{'='*60}")
    print(f"Visualization complete.")
    print(f"  Prediction   : {OUT_DIR}/pred_vs_gt.mp4")
    print(f"  Checkpoint   : {OUT_DIR}/best_model.pt")
    print(f"{'='*60}\n")
