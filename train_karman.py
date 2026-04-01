"""
train_karman.py
===============
Train PDE-Transformer (MC-S) on the Karman vortex CFD simulations under SIM_ROOT.

Each sim_* folder is treated as one independent simulation. Frames are never mixed
across simulations for train/validation splits or for visualization.
"""

import glob
import math
import os
import subprocess
import sys
from contextlib import suppress
from io import BytesIO
from multiprocessing import cpu_count, freeze_support

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import transformers


if os.name != "nt":
    os.environ["LD_LIBRARY_PATH"] = (
        "/home/vatani/miniconda/envs/ag_env/lib/python3.11/site-packages/nvidia/cudnn/lib:" +
        os.environ.get("LD_LIBRARY_PATH", "")
    )

torch.backends.cudnn.enabled = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

torch.cuda.empty_cache()

try:
    import torch.nn.attention.flex_attention

    if not hasattr(torch.nn.attention.flex_attention, "AuxRequest"):
        class AuxRequest:
            pass

        torch.nn.attention.flex_attention.AuxRequest = AuxRequest
except (ImportError, AttributeError):
    pass

if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(*args, **kwargs):
        return [], []

    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


torch.manual_seed(42)
np.random.seed(42)


# User-editable configuration.
# Use normal Python values here instead of environment variables.
SIM_ROOT = "256_inc"  # e.g. r"D:\data\256_inc" or "/data/256_inc"
OUT_DIR = os.path.join("runs", "karman")
EPOCHS = 30
BATCH_SIZE = 32
ACCUM_GRAD = 1
LR = 1e-5
VAL_FRAC = 0.10
WARMUP_FRAC = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS_VID = 10
VID_FRAMES = 50
DPI_VID = 110
SKIP_TRAIN = False
RESUME_CHECKPOINT = "./runs/karman/last.ckpt"  # e.g. r"D:\runs\karman\last.ckpt" or "/home/user/last.ckpt"
MODEL_TYPE = "PDE-S"  # Smallest PDETransformer variant in this repo
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
NUM_WORKERS = max(0, cpu_count() - 1)
PIN_MEMORY = DEVICE == "cuda"


def load_npz_array(path):
    with np.load(path) as data:
        return data["arr_0"].astype(np.float32)


def discover_simulations(root_dir):
    sim_dirs = sorted(
        path for path in glob.glob(os.path.join(root_dir, "sim_*"))
        if os.path.isdir(path)
    )
    sim_infos = []
    for sim_dir in sim_dirs:
        vel_files = sorted(glob.glob(os.path.join(sim_dir, "velocity_*.npz")))
        pre_files = sorted(glob.glob(os.path.join(sim_dir, "pressure_*.npz")))
        if not vel_files or not pre_files:
            continue
        if len(vel_files) != len(pre_files):
            raise RuntimeError(
                f"Frame count mismatch in {sim_dir}: vel={len(vel_files)} pre={len(pre_files)}"
            )
        sim_infos.append(
            {
                "dir": sim_dir,
                "vel": vel_files,
                "pre": pre_files,
                "mask_path": os.path.join(sim_dir, "obstacle_mask.npz"),
                "n_frames": len(vel_files),
            }
        )

    if not sim_infos:
        raise RuntimeError(f"No valid simulation folders found under {root_dir}")

    return sim_infos


def split_simulations(sim_infos, val_frac):
    if len(sim_infos) <= 1:
        return sim_infos, []

    val_count = max(1, int(round(len(sim_infos) * val_frac)))
    if val_count >= len(sim_infos):
        val_count = len(sim_infos) - 1

    return sim_infos[:-val_count], sim_infos[-val_count:]


def warmup_start_index(num_frames, warmup_frac=WARMUP_FRAC):
    return int(num_frames * warmup_frac)


def compute_global_stats(train_sim_infos, fallback_sim_infos, target_samples=200):
    source_sims = train_sim_infos if train_sim_infos else fallback_sim_infos
    total_frames = sum(sim["n_frames"] for sim in source_sims)
    samples = []

    for sim in source_sims:
        start_idx = warmup_start_index(sim["n_frames"])
        usable_frames = max(1, sim["n_frames"] - start_idx)
        sample_count = max(1, int(round(target_samples * (usable_frames / total_frames))))
        idxs = np.linspace(start_idx, sim["n_frames"] - 1, sample_count, dtype=int)
        for idx in idxs:
            v = load_npz_array(sim["vel"][idx])
            p = load_npz_array(sim["pre"][idx])
            samples.append(np.concatenate([v, p], axis=0))

    if not samples:
        sim = source_sims[0]
        v = load_npz_array(sim["vel"][0])
        p = load_npz_array(sim["pre"][0])
        samples.append(np.concatenate([v, p], axis=0))

    stacked = np.stack(samples)
    mean = torch.tensor(stacked.mean(axis=(0, 2, 3)), dtype=torch.float32)
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + 1e-6
    return mean, std


class MultiSimKarmanDataset(Dataset):
    def __init__(self, sim_list, mean, std):
        self.sims = sim_list
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self.samples = []

        for sim_idx, sim in enumerate(self.sims):
            start_idx = warmup_start_index(sim["n_frames"])
            if os.path.exists(sim["mask_path"]):
                mask = load_npz_array(sim["mask_path"])
            else:
                first_frame = load_npz_array(sim["vel"][0])
                mask = np.ones((first_frame.shape[1], first_frame.shape[2]), dtype=np.float32)
            sim["mask_t"] = torch.from_numpy(mask[None, ...]).float()

            for frame_idx in range(start_idx, sim["n_frames"] - 1):
                self.samples.append((sim_idx, frame_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_idx, frame_idx = self.samples[idx]
        sim = self.sims[sim_idx]

        v0 = load_npz_array(sim["vel"][frame_idx])
        p0 = load_npz_array(sim["pre"][frame_idx])
        v1 = load_npz_array(sim["vel"][frame_idx + 1])
        p1 = load_npz_array(sim["pre"][frame_idx + 1])

        x = torch.from_numpy(np.concatenate([v0, p0], axis=0))
        y = torch.from_numpy(np.concatenate([v1, p1], axis=0))

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        return x.float(), y.float(), sim["mask_t"]


def build_loader(dataset, shuffle):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def get_model():
    from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

    return PDETransformer(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        type=MODEL_TYPE,
        patch_size=4,
        periodic=False,
        carrier_token_active=True,
    ).to(DEVICE)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}/")

    print("\nDiscovering simulation folders …")
    sim_infos = discover_simulations(SIM_ROOT)
    print(f"Found {len(sim_infos)} simulations under {SIM_ROOT}")
    total_frames = sum(sim["n_frames"] for sim in sim_infos)
    print(f"Total frames across all simulations: {total_frames}")

    train_sim_infos, val_sim_infos = split_simulations(sim_infos, VAL_FRAC)
    print(f"Train simulations: {len(train_sim_infos)}   Val simulations: {len(val_sim_infos)}")

    print("Computing normalization statistics …")
    mean, std = compute_global_stats(train_sim_infos, sim_infos)
    print(f"  per-channel mean: {mean.numpy().round(5)}")
    print(f"  per-channel std:  {std.numpy().round(5)}")

    train_ds = MultiSimKarmanDataset(train_sim_infos, mean, std)
    val_ds = MultiSimKarmanDataset(val_sim_infos, mean, std) if val_sim_infos else None
    train_loader = build_loader(train_ds, shuffle=True)
    val_loader = build_loader(val_ds, shuffle=False) if val_ds is not None else None
    print(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds) if val_ds is not None else 0}")

    zero_norm = ((torch.zeros(3, device=DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)).view(1, 3, 1, 1)

    print("\nBuilding PDE-Transformer (MC-S) …")
    model = get_model()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")

    adamw_kwargs = {
        "lr": LR,
        "weight_decay": 1e-2,
        "foreach": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        with suppress(TypeError, RuntimeError):
            adamw_kwargs["fused"] = True
            optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        if "optimizer" not in locals():
            adamw_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    task_label = torch.tensor([1000], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    train_losses = []
    val_losses = []
    best_val = math.inf
    start_epoch = 1

    def build_checkpoint(epoch, best_metric):
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_metric,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

    def save_checkpoint(epoch, best_metric):
        torch.save(build_checkpoint(epoch, best_metric), os.path.join(OUT_DIR, "last.ckpt"))

    def load_checkpoint(checkpoint_path):
        nonlocal best_val, start_epoch, train_losses, val_losses
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            saved_epoch = int(checkpoint.get("epoch", 0))
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val = float(checkpoint.get("best_val", best_val))
            train_losses = list(checkpoint.get("train_losses", []))
            val_losses = list(checkpoint.get("val_losses", []))
            start_epoch = saved_epoch + 1
            return True, saved_epoch

        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            return False, None

        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    def save_rollout_video(model_to_render, out_path, title_tag):
        video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
        warmup_idx = warmup_start_index(video_sim["n_frames"])
        usable_len = max(1, video_sim["n_frames"] - warmup_idx)
        start_idx = warmup_idx + int(usable_len * 0.50)
        end_idx = warmup_idx + int(usable_len * 0.75)
        end_idx = min(video_sim["n_frames"], max(start_idx + 1, end_idx))
        rollout_len = end_idx - start_idx

        gt_frames = []
        for idx in range(start_idx, end_idx):
            v = load_npz_array(video_sim["vel"][idx])
            p = load_npz_array(video_sim["pre"][idx])
            gt_frames.append(np.concatenate([v, p], axis=0))
        gt_frames = np.stack(gt_frames)

        mean_np = mean.numpy()[:, None, None]
        std_np = std.numpy()[:, None, None]
        gt_norm = (gt_frames - mean_np) / std_np

        if os.path.exists(video_sim["mask_path"]):
            mask_np = load_npz_array(video_sim["mask_path"])
        else:
            sample = load_npz_array(video_sim["vel"][0])
            mask_np = np.ones((sample.shape[1], sample.shape[2]), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np[None, ...]).float().to(DEVICE).unsqueeze(0)

        pred_frames = [gt_norm[0]]
        model_to_render.eval()
        with torch.inference_mode():
            current = torch.tensor(gt_norm[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            labels = get_labels(1)
            for _ in range(rollout_len - 1):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    nxt = model_to_render(current, class_labels=labels).sample
                nxt = nxt.float()
                nxt = torch.lerp(zero_norm, nxt, mask_tensor)
                pred_frames.append(nxt[0].cpu().numpy())
                current = nxt

        pred_frames = np.stack(pred_frames)
        gt_norm_seq = gt_norm[: len(pred_frames)]

        def unnorm(frame, channel):
            return frame[channel] * std_np[channel, 0, 0] + mean_np[channel, 0, 0]

        def vel_mag(frame):
            vx = unnorm(frame, 0)
            vy = unnorm(frame, 1)
            return np.sqrt(vx**2 + vy**2)

        def pres(frame):
            return unnorm(frame, 2)

        fig_w, fig_h = 12, 5.5
        cmap_vel = "viridis"
        cmap_pres = "RdBu_r"
        gt_vel_max = np.percentile([vel_mag(frame) for frame in gt_norm_seq], 100)
        gt_pres_all = np.concatenate([pres(frame).ravel() for frame in gt_norm_seq])
        gt_pres_abs = np.percentile(np.abs(gt_pres_all), 100)

        norm_vel = Normalize(vmin=0, vmax=gt_vel_max)
        norm_pres = Normalize(vmin=-gt_pres_abs, vmax=gt_pres_abs)

        writer = None
        writer_mode = "gif"
        try:
            writer = animation.FFMpegWriter(fps=FPS_VID, codec="libx264", bitrate=1800)
            writer_mode = "mp4"
        except (RuntimeError, FileNotFoundError, OSError):
            writer = animation.PillowWriter(fps=FPS_VID)

        def render_frame(idx, fig):
            fig.clf()
            gs = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.95, top=0.88, bottom=0.08, hspace=0.45, wspace=0.10)
            axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

            titles = ["GT  |u|", "Pred  |u|", "GT  p", "Pred  p"]
            data = [vel_mag(gt_norm_seq[idx]), vel_mag(pred_frames[idx]), pres(gt_norm_seq[idx]), pres(pred_frames[idx])]
            cmaps = [cmap_vel, cmap_vel, cmap_pres, cmap_pres]
            norms = [norm_vel, norm_vel, norm_pres, norm_pres]

            for ax, image, title, cmap, norm in zip(axes, data, titles, cmaps, norms):
                ax.set_facecolor("#111")
                im = ax.imshow(np.rot90(image, k=1), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
                ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=3)
                ax.axis("off")
                cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, format="%.3f")
                cb.ax.tick_params(colors="#ccc", labelsize=6)
                cb.outline.set_edgecolor("#555")

            fig.text(
                0.5,
                0.94,
                f"{title_tag}   |   t = {start_idx + idx:05d}   |   Frame {idx:03d}/{len(pred_frames) - 1}",
                ha="center",
                va="top",
                color="white",
                fontsize=9,
                fontfamily="monospace",
            )

        fig = plt.figure(figsize=(12, 5.5), dpi=DPI_VID, facecolor="#111")
        try:
            with writer.saving(fig, out_path, dpi=DPI_VID):
                for idx in tqdm(range(len(pred_frames)), desc=f"Rendering {os.path.basename(out_path)}"):
                    render_frame(idx, fig)
                    writer.grab_frame(facecolor=fig.get_facecolor())
        except (RuntimeError, FileNotFoundError, OSError):
            if writer_mode == "mp4":
                gif_path = os.path.splitext(out_path)[0] + ".gif"
                writer = animation.PillowWriter(fps=FPS_VID)
                with writer.saving(fig, gif_path, dpi=DPI_VID):
                    for idx in tqdm(range(len(pred_frames)), desc=f"Rendering {os.path.basename(gif_path)}"):
                        render_frame(idx, fig)
                        writer.grab_frame(facecolor=fig.get_facecolor())
                out_path = gif_path
            else:
                raise
        finally:
            plt.close(fig)

        print(f"  Saved: {out_path}")
        return out_path

    resume_path = RESUME_CHECKPOINT
    if resume_path and os.path.exists(resume_path):
        resumed, saved_epoch = load_checkpoint(resume_path)
        print(f"Loaded checkpoint: {resume_path}")
        if resumed:
            print(
                f"Successfully loaded checkpoint weights. "
                f"Checkpoint was saved at epoch {saved_epoch}; resuming at epoch {start_epoch}."
            )
        else:
            print("Successfully loaded model weights from checkpoint; optimizer/scheduler state not found.")

    if not SKIP_TRAIN:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            epoch_train_loss = 0.0
            n_train = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (x, y, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [train]", leave=False)):
                x = x.to(DEVICE, non_blocking=PIN_MEMORY)
                y = y.to(DEVICE, non_blocking=PIN_MEMORY)
                mask = mask.to(DEVICE, non_blocking=PIN_MEMORY)
                labels = get_labels(x.shape[0])

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(x, class_labels=labels).sample
                pred = pred.float()
                pred = torch.lerp(zero_norm, pred, mask)

                raw_loss = F.mse_loss(pred, y)
                (raw_loss / ACCUM_GRAD).backward()

                if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_train_loss += raw_loss.item() * x.shape[0]
                n_train += x.shape[0]

            train_loss = epoch_train_loss / max(1, n_train)

            if val_loader is None:
                val_loss = float("inf")
            else:
                model.eval()
                epoch_val_loss = 0.0
                n_val = 0
                with torch.inference_mode():
                    for x, y, mask in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [val]  ", leave=False):
                        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
                        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
                        mask = mask.to(DEVICE, non_blocking=PIN_MEMORY)
                        labels = get_labels(x.shape[0])
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                            pred = model(x, class_labels=labels).sample
                        pred = pred.float()
                        pred = torch.lerp(zero_norm, pred, mask)
                        raw_loss = F.mse_loss(pred, y)
                        epoch_val_loss += raw_loss.item() * x.shape[0]
                        n_val += x.shape[0]

                val_loss = epoch_val_loss / max(1, n_val)

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_marker = " ← best"
            else:
                best_marker = ""

            save_checkpoint(epoch, best_val)

            if epoch % 2 == 0:
                epoch_video_path = os.path.join(OUT_DIR, f"pred_vs_gt_epoch_{epoch:03d}.mp4")
                save_rollout_video(model, epoch_video_path, f"Epoch {epoch:03d}")

            print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{best_marker}")

        print("\nSaving loss curve …")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#111")
        ax.set_facecolor("#111")
        ax.plot(range(1, EPOCHS + 1), train_losses, color="#00c8ff", linewidth=2, label="Train MSE")
        ax.plot(range(1, EPOCHS + 1), val_losses, color="#ff6b35", linewidth=2, label="Val MSE")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("MSE Loss", color="white")
        ax.set_title("PDE-Transformer - Karman Vortex Street", color="white", fontsize=13)
        ax.legend(framealpha=0.3)
        ax.tick_params(colors="white")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        plt.tight_layout()
        curve_path = os.path.join(OUT_DIR, "loss_curve.png")
        plt.savefig(curve_path, dpi=150, facecolor="#111")
        plt.close()
        print(f"  Saved: {curve_path}")
    else:
        print("\nSkipping training (SKIP_TRAIN=True).")

    print("\nGenerating prediction vs ground-truth rollout video …")

    best_path = os.path.join(OUT_DIR, "last.ckpt")
    if not os.path.exists(best_path):
        print("No last.ckpt found, skipping video.")
        return

    checkpoint = torch.load(best_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    frame_path = save_rollout_video(model, os.path.join(OUT_DIR, "pred_vs_gt.mp4"), "Final")
    if not SKIP_TRAIN:
        print(f"\n{'=' * 60}")
        print("Training complete.")
        print(f"  Best val MSE : {best_val:.6f}")
        print(f"  Loss curve   : {OUT_DIR}/loss_curve.png")
        print(f"  Prediction   : {frame_path}")
        print(f"  Checkpoint   : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 60}\n")
        print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}")
        print("-" * 35)
        for epoch_idx, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            print(f"{epoch_idx:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}")
    else:
        print(f"\n{'=' * 60}")
        print("Visualization complete.")
        print(f"  Prediction   : {frame_path}")
        print(f"  Checkpoint   : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)