"""
loss_evidance.py
================
Autoregressive long-rollout validation to quantify why field-only MSE training degrades
for long horizons.

What this script does:
1. Loads a trained PDE surrogate checkpoint (default: runs/karman/last_mse_only.ckpt).
2. Runs a true autoregressive rollout on sim_000082 starting at 50% of the sequence.
3. Predicts 291 steps (or as many as available) with prediction feedback at each step.
4. Computes spatial error maps for:
   - MSE (field-space squared error)
   - Gradient magnitude mismatch
   - Gradient directional mismatch
5. Saves:
   - GIF of spatial error contours (plasma colormap)
   - Per-step CSV and per-step distribution CSV
   - Statistical summary CSV + text report
   - Plot panel with curves and numerical second derivative of MSE vs step
"""

from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import transformers
from sim_cache import discover_simulations, ensure_all_sim_caches, load_packed_array


# Compatibility patches mirrored from training script for stable model import/loading.
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


# -------------------------
# User-editable parameters.
# -------------------------
SIM_ROOT = "/home/vatani/data_vortex/256_inc"
TARGET_SIM_NAME = "sim_000082"
CHECKPOINT_PATH = "runs/karman/last_mse_only.ckpt"
OUT_DIR = "runs/error_comparison/loss_evidence_MSE"


MODEL_TYPE = "PDE-S"
WARMUP_FRAC = 0.50
VAL_FRAC = 0.10
ROLLOUT_STEPS = 291

CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
MP_WORKERS = 6
CACHE_WORKERS = MP_WORKERS

DIRECTION_EPS = 1e-8
DIRECTION_TAU = 1e-4

GIF_FPS = 10
GIF_DPI = 180
GIF_NAME = "error_contours_mse_grad_direction.gif"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def warmup_start_index(num_frames: int, warmup_frac: float = WARMUP_FRAC) -> int:
    return int(num_frames * warmup_frac)


def split_simulations(sim_infos: List[dict], val_frac: float) -> Tuple[List[dict], List[dict]]:
    if len(sim_infos) <= 1:
        return sim_infos, []

    val_count = max(1, int(round(len(sim_infos) * val_frac)))
    if val_count >= len(sim_infos):
        val_count = len(sim_infos) - 1

    return sim_infos[:-val_count], sim_infos[-val_count:]


def _sample_sim_for_stats(sim: dict, total_frames: int, target_samples: int) -> np.ndarray:
    states = load_packed_array(sim["states_path"])
    start_idx = warmup_start_index(sim["n_frames"])
    usable_frames = max(1, sim["n_frames"] - start_idx)
    sample_count = max(1, int(round(target_samples * (usable_frames / max(1, total_frames)))))
    idxs = np.linspace(start_idx, sim["n_frames"] - 1, sample_count, dtype=int)
    return np.asarray(states[idxs], dtype=np.float32)


def compute_global_stats(train_sim_infos: List[dict], fallback_sim_infos: List[dict], target_samples: int = 200):
    source_sims = train_sim_infos if train_sim_infos else fallback_sim_infos
    total_frames = sum(sim["n_frames"] for sim in source_sims)
    samples = []

    if source_sims:
        try:
            with ProcessPoolExecutor(max_workers=MP_WORKERS) as pool:
                futures = [
                    pool.submit(_sample_sim_for_stats, sim, total_frames, target_samples)
                    for sim in source_sims
                ]
                for fut in futures:
                    samples.append(fut.result())
        except Exception:
            # Fallback to serial sampling if multiprocessing is unavailable.
            for sim in source_sims:
                samples.append(_sample_sim_for_stats(sim, total_frames, target_samples))

    if not samples:
        sim = source_sims[0]
        states = load_packed_array(sim["states_path"])
        samples.append(np.asarray(states[[0]], dtype=np.float32))

    stacked = np.concatenate(samples, axis=0)
    mean = torch.tensor(stacked.mean(axis=(0, 2, 3)), dtype=torch.float32)
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + 1e-6
    return mean, std


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


def load_checkpoint_weights(model: torch.nn.Module, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")


class SpatialErrorComputer:
    def __init__(self, eps: float = DIRECTION_EPS, tau: float = DIRECTION_TAU):
        self.eps = float(eps)
        self.tau = float(tau)

    def _grads_2d(self, x: torch.Tensor):
        gy = torch.empty_like(x)
        gx = torch.empty_like(x)

        gy[:, :, 1:-1, :] = 0.5 * (x[:, :, 2:, :] - x[:, :, :-2, :])
        gy[:, :, 0, :] = x[:, :, 1, :] - x[:, :, 0, :]
        gy[:, :, -1, :] = x[:, :, -1, :] - x[:, :, -2, :]

        gx[:, :, :, 1:-1] = 0.5 * (x[:, :, :, 2:] - x[:, :, :, :-2])
        gx[:, :, :, 0] = x[:, :, :, 1] - x[:, :, :, 0]
        gx[:, :, :, -1] = x[:, :, :, -1] - x[:, :, :, -2]

        return gy, gx

    def maps(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, np.ndarray]:
        # Inputs are [1, C, H, W].
        sq = (pred - target).square()  # [1, C, H, W]
        mse_map = sq.mean(dim=1, keepdim=False)  # [1, H, W]

        pred_gy, pred_gx = self._grads_2d(pred)
        tgt_gy, tgt_gx = self._grads_2d(target)

        pred_g = torch.stack((pred_gy, pred_gx), dim=2)  # [1, C, 2, H, W]
        tgt_g = torch.stack((tgt_gy, tgt_gx), dim=2)

        grad_diff_sq = (pred_g - tgt_g).square().sum(dim=2)  # [1, C, H, W]
        grad_map = grad_diff_sq.mean(dim=1, keepdim=False)   # [1, H, W]

        dot = (pred_g * tgt_g).sum(dim=2)  # [1, C, H, W]
        pred_norm = torch.sqrt(pred_g.square().sum(dim=2) + self.eps)
        tgt_norm = torch.sqrt(tgt_g.square().sum(dim=2) + self.eps)
        cos = dot / (pred_norm * tgt_norm + self.eps)
        dir_loss_pointwise = 1.0 - cos

        w_dir = tgt_norm / (tgt_norm + self.tau)
        dir_map = (dir_loss_pointwise * w_dir).mean(dim=1, keepdim=False)  # [1, H, W]

        # Mask invalid/solid regions so domain stats represent fluid interior.
        m = mask
        if m.ndim == 4 and m.shape[1] != 1:
            m = m.mean(dim=1, keepdim=False)
        elif m.ndim == 4:
            m = m[:, 0]
        else:
            raise ValueError(f"Unexpected mask shape: {m.shape}")
        m = m.clamp(0.0, 1.0)

        mse_map = (mse_map * m).squeeze(0).detach().cpu().numpy()
        grad_map = (grad_map * m).squeeze(0).detach().cpu().numpy()
        dir_map = (dir_map * m).squeeze(0).detach().cpu().numpy()
        m_np = m.squeeze(0).detach().cpu().numpy()

        return {
            "mse": mse_map,
            "grad": grad_map,
            "direction": dir_map,
            "mask": m_np,
        }


def scalar_from_map(err_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = mask > 0.5
    vals = err_map[valid]
    if vals.size == 0:
        vals = err_map.ravel()

    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def finite_diff_first(y: np.ndarray) -> np.ndarray:
    # Unit step spacing between rollout indices.
    return np.gradient(y)


def finite_diff_second(y: np.ndarray) -> np.ndarray:
    return np.gradient(np.gradient(y))


def make_error_gif(
    vel_maps: np.ndarray,
    mse_maps: np.ndarray,
    grad_maps: np.ndarray,
    dir_maps: np.ndarray,
    mask_maps: np.ndarray,
    out_path: str,
    start_idx: int,
):
    t_steps = mse_maps.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), dpi=GIF_DPI, facecolor="white")
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.10, top=0.86, wspace=0.18)

    for ax in axes:
        ax.set_facecolor("white")
        ax.axis("off")

    vmax_vel = float(np.percentile(vel_maps, 99.0)) + 1e-12
    # Robust scaling keeps visualization readable as errors blow up.
    vmax_mse = float(np.percentile(mse_maps, 99.0)) + 1e-12
    vmax_grad = float(np.percentile(grad_maps, 99.0)) + 1e-12
    vmax_dir = float(np.percentile(dir_maps, 99.0)) + 1e-12

    cmap_vel = plt.cm.viridis.copy()
    cmap_vel.set_bad(color="white")
    cmap_mse = plt.cm.plasma.copy()
    cmap_grad = plt.cm.plasma.copy()
    cmap_dir = plt.cm.plasma.copy()
    for cmap in (cmap_mse, cmap_grad, cmap_dir):
        cmap.set_bad(color="white")

    mask0 = mask_maps[0] > 0.5
    mask0_rot = np.rot90(mask0, k=3)
    im_vel = axes[0].imshow(
        np.ma.masked_where(~mask0_rot, np.rot90(vel_maps[0], k=3)),
        cmap=cmap_vel,
        origin="lower",
        vmin=0.0,
        vmax=vmax_vel,
        interpolation="lanczos",
    )
    im0 = axes[1].imshow(
        np.ma.masked_where(~mask0_rot, np.rot90(mse_maps[0], k=3)),
        cmap=cmap_mse,
        origin="lower",
        vmin=0.0,
        vmax=vmax_mse,
        interpolation="lanczos",
    )
    im1 = axes[2].imshow(
        np.ma.masked_where(~mask0_rot, np.rot90(grad_maps[0], k=3)),
        cmap=cmap_grad,
        origin="lower",
        vmin=0.0,
        vmax=vmax_grad,
        interpolation="lanczos",
    )
    im2 = axes[3].imshow(
        np.ma.masked_where(~mask0_rot, np.rot90(dir_maps[0], k=3)),
        cmap=cmap_dir,
        origin="lower",
        vmin=0.0,
        vmax=vmax_dir,
        interpolation="lanczos",
    )

    title_size = 15
    suptitle_size = 17
    cbar_tick_size = 10

    axes[0].set_title("Velocity Field", color="black", fontsize=title_size)
    axes[1].set_title("MSE Error Map", color="black", fontsize=title_size)
    axes[2].set_title("Gradient Error Map", color="black", fontsize=title_size)
    axes[3].set_title("Directional Error Map", color="black", fontsize=title_size)

    c0 = fig.colorbar(im_vel, ax=axes[0], orientation="horizontal", fraction=0.05, pad=0.08)
    c1 = fig.colorbar(im0, ax=axes[1], orientation="horizontal", fraction=0.05, pad=0.08)
    c2 = fig.colorbar(im1, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.08)
    c3 = fig.colorbar(im2, ax=axes[3], orientation="horizontal", fraction=0.05, pad=0.08)

    for cb in (c0, c1, c2, c3):
        cb.ax.tick_params(colors="black", labelsize=cbar_tick_size)
        cb.outline.set_edgecolor("#777")

    supt = fig.suptitle("", color="black", fontsize=suptitle_size, y=0.965)

    def _update(frame_idx: int):
        mask = np.rot90(mask_maps[frame_idx] > 0.5, k=3)
        im_vel.set_data(np.ma.masked_where(~mask, np.rot90(vel_maps[frame_idx], k=3)))
        im0.set_data(np.ma.masked_where(~mask, np.rot90(mse_maps[frame_idx], k=3)))
        im1.set_data(np.ma.masked_where(~mask, np.rot90(grad_maps[frame_idx], k=3)))
        im2.set_data(np.ma.masked_where(~mask, np.rot90(dir_maps[frame_idx], k=3)))
        supt.set_text(
            f"Autoregressive Rollout Error Contours | step={frame_idx + 1:03d}/{t_steps:03d} | "
            f"global t={start_idx + frame_idx + 1:05d}"
        )
        return im_vel, im0, im1, im2, supt

    ani = animation.FuncAnimation(fig, _update, frames=t_steps, interval=1000.0 / GIF_FPS, blit=False)
    writer = animation.PillowWriter(fps=GIF_FPS)
    ani.save(out_path, writer=writer)
    plt.close(fig)


def save_step_csv(rows: List[Dict[str, float]], out_csv: str):
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_summary(step_rows: List[Dict[str, float]], summary_csv: str, summary_txt: str):
    metrics = {
        "mse_mean": np.array([r["mse_mean"] for r in step_rows], dtype=np.float64),
        "grad_mean": np.array([r["grad_mean"] for r in step_rows], dtype=np.float64),
        "direction_mean": np.array([r["direction_mean"] for r in step_rows], dtype=np.float64),
    }

    x = np.arange(len(step_rows), dtype=np.float64)
    summary_rows = []

    for name, arr in metrics.items():
        slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) > 1 else float("nan")
        auc = float(np.trapezoid(arr, x=x)) if len(arr) > 1 else float(arr[0])
        summary_rows.append(
            {
                "metric": name,
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p95": float(np.percentile(arr, 95)),
                "slope_per_step": slope,
                "auc": auc,
            }
        )

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    corr_mg = float(np.corrcoef(metrics["mse_mean"], metrics["grad_mean"])[0, 1])
    corr_md = float(np.corrcoef(metrics["mse_mean"], metrics["direction_mean"])[0, 1])
    corr_gd = float(np.corrcoef(metrics["grad_mean"], metrics["direction_mean"])[0, 1])

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Rollout Statistical Study\n")
        f.write("=" * 80 + "\n")
        f.write(f"Steps: {len(step_rows)}\n")
        f.write("\nPer-metric summary:\n")
        for r in summary_rows:
            f.write(
                f"- {r['metric']}: mean={r['mean']:.6e}, std={r['std']:.6e}, "
                f"median={r['median']:.6e}, min={r['min']:.6e}, max={r['max']:.6e}, "
                f"p95={r['p95']:.6e}, slope={r['slope_per_step']:.6e}, auc={r['auc']:.6e}\n"
            )
        f.write("\nCross-metric Pearson correlations:\n")
        f.write(f"- corr(mse, grad)      = {corr_mg:.6f}\n")
        f.write(f"- corr(mse, direction) = {corr_md:.6f}\n")
        f.write(f"- corr(grad, direction)= {corr_gd:.6f}\n")


@dataclass
class RolloutResults:
    step_rows: List[Dict[str, float]]
    vel_maps: np.ndarray
    pred_vel_uv: np.ndarray
    mse_maps: np.ndarray
    grad_maps: np.ndarray
    dir_maps: np.ndarray
    mask_maps: np.ndarray


def run_rollout(
    model: torch.nn.Module,
    states: np.ndarray,
    mask_tensor: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    start_idx: int,
    num_steps: int,
) -> RolloutResults:
    err_comp = SpatialErrorComputer()

    mean_b = mean.to(DEVICE).view(1, 3, 1, 1)
    std_b = std.to(DEVICE).view(1, 3, 1, 1)
    zero_norm = ((torch.zeros(3, device=DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)).view(1, 3, 1, 1)

    labels = torch.tensor([1000], dtype=torch.long, device=DEVICE)

    # Normalize full sequence once for stable target extraction.
    gt = torch.from_numpy(states.astype(np.float32)).to(DEVICE)
    gt_norm = (gt - mean_b) / std_b

    current = gt_norm[start_idx : start_idx + 1].clone()  # first input is ground truth

    rows: List[Dict[str, float]] = []
    vel_maps: List[np.ndarray] = []
    pred_vel_uv: List[np.ndarray] = []
    mse_maps: List[np.ndarray] = []
    grad_maps: List[np.ndarray] = []
    dir_maps: List[np.ndarray] = []
    mask_maps: List[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for step in range(num_steps):
            target = gt_norm[start_idx + step + 1 : start_idx + step + 2]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                pred = model(current, class_labels=labels).sample
            pred = pred.float()
            pred = torch.lerp(zero_norm, pred, mask_tensor)

            maps = err_comp.maps(pred, target, mask_tensor)
            mse_map = maps["mse"]
            grad_map = maps["grad"]
            dir_map = maps["direction"]
            mask_np = maps["mask"]
            pred_phys = pred * std_b + mean_b
            pred_uv = pred_phys[0, :2].detach().cpu().numpy().astype(np.float32)
            vel_map = torch.sqrt(pred_phys[0, 0].square() + pred_phys[0, 1].square()).detach().cpu().numpy()

            mse_s = scalar_from_map(mse_map, mask_np)
            grad_s = scalar_from_map(grad_map, mask_np)
            dir_s = scalar_from_map(dir_map, mask_np)

            rows.append(
                {
                    "step": float(step + 1),
                    "global_frame": float(start_idx + step + 1),
                    "mse_mean": mse_s["mean"],
                    "mse_std": mse_s["std"],
                    "mse_median": mse_s["median"],
                    "mse_p90": mse_s["p90"],
                    "mse_p95": mse_s["p95"],
                    "mse_min": mse_s["min"],
                    "mse_max": mse_s["max"],
                    "grad_mean": grad_s["mean"],
                    "grad_std": grad_s["std"],
                    "grad_median": grad_s["median"],
                    "grad_p90": grad_s["p90"],
                    "grad_p95": grad_s["p95"],
                    "grad_min": grad_s["min"],
                    "grad_max": grad_s["max"],
                    "direction_mean": dir_s["mean"],
                    "direction_std": dir_s["std"],
                    "direction_median": dir_s["median"],
                    "direction_p90": dir_s["p90"],
                    "direction_p95": dir_s["p95"],
                    "direction_min": dir_s["min"],
                    "direction_max": dir_s["max"],
                }
            )

            mse_maps.append(mse_map)
            grad_maps.append(grad_map)
            dir_maps.append(dir_map)
            vel_maps.append(vel_map * mask_np)
            pred_vel_uv.append(pred_uv)
            mask_maps.append(mask_np)

            # True autoregressive unrolling: feed back prediction.
            current = pred

    # Add numerical derivatives to per-step table.
    mse_curve = np.array([r["mse_mean"] for r in rows], dtype=np.float64)
    d1 = finite_diff_first(mse_curve)
    d2 = finite_diff_second(mse_curve)
    for i, r in enumerate(rows):
        r["mse_d1"] = float(d1[i])
        r["mse_d2"] = float(d2[i])

    return RolloutResults(
        step_rows=rows,
        vel_maps=np.stack(vel_maps, axis=0),
        pred_vel_uv=np.stack(pred_vel_uv, axis=0),
        mse_maps=np.stack(mse_maps, axis=0),
        grad_maps=np.stack(grad_maps, axis=0),
        dir_maps=np.stack(dir_maps, axis=0),
        mask_maps=np.stack(mask_maps, axis=0),
    )


def save_study_plot(step_rows: List[Dict[str, float]], out_png: str):
    step = np.array([r["step"] for r in step_rows], dtype=np.float64)
    mse = np.array([r["mse_mean"] for r in step_rows], dtype=np.float64)
    grad = np.array([r["grad_mean"] for r in step_rows], dtype=np.float64)
    dire = np.array([r["direction_mean"] for r in step_rows], dtype=np.float64)
    d1 = np.array([r["mse_d1"] for r in step_rows], dtype=np.float64)
    d2 = np.array([r["mse_d2"] for r in step_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200, facecolor="white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=12)
        for spine in ax.spines.values():
            spine.set_edgecolor("#777")

    ax = axes[0, 0]
    ax.plot(step, mse, color="#00d5ff", lw=2, label="MSE")
    ax.plot(step, grad, color="#ff8f00", lw=2, label="Gradient")
    ax.plot(step, dire, color="#7bff00", lw=2, label="Directional")
    ax.set_title("Mean Error vs Step", color="black", fontsize=15)
    ax.set_xlabel("Rollout step", color="black", fontsize=13)
    ax.set_ylabel("Error", color="black", fontsize=13)
    ax.set_yscale("log")
    ax.legend(framealpha=0.25, fontsize=11)

    ax = axes[0, 1]
    ax.plot(step, mse, color="#00d5ff", lw=1.8, label="MSE")
    ax.set_title("MSE Trend", color="black", fontsize=15)
    ax.set_xlabel("Rollout step", color="black", fontsize=13)
    ax.set_ylabel("MSE", color="black", fontsize=13)
    ax.legend(framealpha=0.25, fontsize=11)

    ax = axes[1, 0]
    ax.plot(step, d1, color="#ffd166", lw=1.8)
    ax.axhline(0.0, color="#888", lw=1.0, ls="--")
    ax.set_title("First Derivative d(MSE)/d(step)", color="black", fontsize=15)
    ax.set_xlabel("Rollout step", color="black", fontsize=13)
    ax.set_ylabel("d1", color="black", fontsize=13)

    ax = axes[1, 1]
    ax.plot(step, d2, color="#ff4d6d", lw=1.8)
    ax.axhline(0.0, color="#888", lw=1.0, ls="--")
    ax.set_title("Second Derivative d2(MSE)/d(step)^2", color="black", fontsize=15)
    ax.set_xlabel("Rollout step", color="black", fontsize=13)
    ax.set_ylabel("d2", color="black", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Discovering simulations under: {SIM_ROOT}")
    sim_infos = discover_simulations(SIM_ROOT)
    ensure_all_sim_caches(sim_infos, CACHE_WORKERS, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)

    train_sim_infos, _ = split_simulations(sim_infos, VAL_FRAC)
    mean, std = compute_global_stats(train_sim_infos, sim_infos)

    target = next((s for s in sim_infos if os.path.basename(s["dir"]) == TARGET_SIM_NAME), None)
    if target is None:
        raise RuntimeError(f"Target simulation {TARGET_SIM_NAME} not found under {SIM_ROOT}")

    print(f"Using target simulation: {target['dir']}")
    states = np.asarray(load_packed_array(target["states_path"]), dtype=np.float32)
    mask_np = np.asarray(load_packed_array(target["packed_mask_path"]), dtype=np.float32)

    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
        mask_t = torch.from_numpy(mask_np).to(DEVICE).unsqueeze(0).float()
    else:
        raise RuntimeError(f"Unexpected mask shape: {mask_np.shape}")

    start_idx = warmup_start_index(target["n_frames"], WARMUP_FRAC)
    max_steps = target["n_frames"] - start_idx - 1
    if max_steps <= 0:
        raise RuntimeError("Not enough frames for rollout after warmup start.")

    steps = min(ROLLOUT_STEPS, max_steps)
    if steps < ROLLOUT_STEPS:
        print(f"Requested {ROLLOUT_STEPS} steps but only {steps} are available; using {steps}.")

    model = get_model()
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint does not exist: {CHECKPOINT_PATH}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    load_checkpoint_weights(model, CHECKPOINT_PATH)

    print(
        f"Running true autoregressive rollout: start_idx={start_idx}, steps={steps}, "
        f"start_frac={WARMUP_FRAC:.2f}"
    )

    results = run_rollout(
        model=model,
        states=states,
        mask_tensor=mask_t,
        mean=mean,
        std=std,
        start_idx=start_idx,
        num_steps=steps,
    )

    step_csv = os.path.join(OUT_DIR, "rollout_step_metrics.csv")
    summary_csv = os.path.join(OUT_DIR, "rollout_metric_summary.csv")
    summary_txt = os.path.join(OUT_DIR, "rollout_statistical_study.txt")
    gif_path = os.path.join(OUT_DIR, GIF_NAME)
    plot_path = os.path.join(OUT_DIR, "rollout_study_plots.png")

    save_step_csv(results.step_rows, step_csv)
    save_metric_summary(results.step_rows, summary_csv, summary_txt)
    save_study_plot(results.step_rows, plot_path)
    make_error_gif(
        results.vel_maps,
        results.mse_maps,
        results.grad_maps,
        results.dir_maps,
        results.mask_maps,
        gif_path,
        start_idx=start_idx,
    )

    # Also save raw map tensors for downstream quantitative analyses.
    np.save(os.path.join(OUT_DIR, "predicted_velocity_uv.npy"), results.pred_vel_uv)
    np.save(os.path.join(OUT_DIR, "predicted_speed_maps.npy"), results.vel_maps)
    np.save(os.path.join(OUT_DIR, "mse_error_maps.npy"), results.mse_maps)
    np.save(os.path.join(OUT_DIR, "grad_error_maps.npy"), results.grad_maps)
    np.save(os.path.join(OUT_DIR, "direction_error_maps.npy"), results.dir_maps)

    print("\nArtifacts saved:")
    print(f"- Step CSV         : {step_csv}")
    print(f"- Summary CSV      : {summary_csv}")
    print(f"- Statistical text : {summary_txt}")
    print(f"- Study plots      : {plot_path}")
    print(f"- Error contour GIF: {gif_path}")
    print(f"- Pred velocity UV : {OUT_DIR}/predicted_velocity_uv.npy")
    print(f"- Pred speed maps  : {OUT_DIR}/predicted_speed_maps.npy")
    print(f"- Raw maps         : {OUT_DIR}/*.npy")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
