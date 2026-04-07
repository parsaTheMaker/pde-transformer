"""
error_comparison.py
===================
Compare rollout error between:
1) Base-trained PDE-Transformer checkpoint.
2) Fine-tuned checkpoint that wraps the latent block with a SequentialModel.

Outputs:
- Per-frame paired metrics CSV.
- Aggregate and step-wise plots.
- Error-map figure using white mask + plasma colormap with shared range.
- Markdown report with statistical comparison and effect-size style summaries.
"""

import argparse
import csv
import math
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# Make fonts larger everywhere (roughly double previous sizes)
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "figure.titlesize": 22,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
import numpy as np
import torch
import torch.nn as nn
import transformers


if not hasattr(transformers.pytorch_utils, "find_pruneable_heads_and_indices"):
    def find_pruneable_heads_and_indices(*args, **kwargs):
        return [], []

    transformers.pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

from pdetransformer.core.sub_network.llm import SequentialModel
from sim_cache import discover_simulations, ensure_all_sim_caches, load_packed_array


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


DEFAULT_SIM_ROOT = "/home/vatani/data_vortex/256_inc"
DEFAULT_BASE_CKPT = "/home/vatani/repos/pde-transformer/runs/karman/last.ckpt"
DEFAULT_FINETUNED_CKPT = "/home/vatani/repos/pde-transformer/runs/karman_finetuned/last.ckpt"
DEFAULT_OUT_ROOT = "/home/vatani/repos/pde-transformer/runs/error_comparison"


MODEL_TYPE = "PDE-S"
VAL_FRAC = 0.10
WARMUP_FRAC = 0.50
ROLLOUT_START_FRAC = 0.50
ROLLOUT_END_FRAC = 0.95

# Fine-tuned sub-network config must match training in fine_tune_karman.py.
SEQ_HIDDEN_SIZE = 144
SEQ_NUM_HEADS = 8
SEQ_N_LAYERS = 4
SEQ_ATTN_METHOD = "naive"


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
        states = load_packed_array(sim["states_path"])
        start_idx = warmup_start_index(sim["n_frames"])
        usable_frames = max(1, sim["n_frames"] - start_idx)
        sample_count = max(1, int(round(target_samples * (usable_frames / max(1, total_frames)))))
        idxs = np.linspace(start_idx, sim["n_frames"] - 1, sample_count, dtype=int)
        samples.append(np.asarray(states[idxs], dtype=np.float32))

    if not samples:
        sim = source_sims[0]
        states = load_packed_array(sim["states_path"])
        samples.append(np.asarray(states[[0]], dtype=np.float32))

    stacked = np.concatenate(samples, axis=0)
    mean = stacked.mean(axis=(0, 2, 3)).astype(np.float32)
    std = stacked.std(axis=(0, 2, 3)).astype(np.float32) + 1e-6
    return mean, std


class LatentWrapper(nn.Module):
    def __init__(self, orig_latent, seq_model):
        super().__init__()
        self.orig_latent = orig_latent
        self.seq_model = seq_model

    def forward(self, x, c):
        x = self.orig_latent(x, c)
        original_dtype = x.dtype
        out = self.seq_model(x.unsqueeze(-1)).squeeze(-1)
        return (x + out).to(original_dtype)


def build_base_model(device):
    from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

    model = PDETransformer(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        type=MODEL_TYPE,
        patch_size=4,
        periodic=False,
        carrier_token_active=True,
    ).to(device)
    return model


def build_finetuned_model(device):
    base_model = build_base_model(device)
    seq_model = SequentialModel(
        in_dim=384,
        mlp_ratio=4,
        use_checkpoint=False,
        hidden_size=SEQ_HIDDEN_SIZE,
        num_heads=SEQ_NUM_HEADS,
        n_layers=SEQ_N_LAYERS,
        attention_method=SEQ_ATTN_METHOD,
        in_context_patches=-1,
        init_zero_proj=True,
        use_conv_proj=False,
    ).to(device)
    base_model.model.latent = LatentWrapper(base_model.model.latent, seq_model)
    return base_model


def load_checkpoint_weights(model, ckpt_path, device, strict=True):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)


def rollout_norm(model, first_frame_norm, mask_t, zero_norm, labels, rollout_len, use_amp):
    model.eval()
    preds = [first_frame_norm]
    current = first_frame_norm.unsqueeze(0)
    with torch.inference_mode():
        for _ in range(rollout_len - 1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                nxt = model(current, class_labels=labels).sample
            nxt = nxt.float()
            nxt = torch.lerp(zero_norm, nxt, mask_t)
            preds.append(nxt[0])
            current = nxt
    return torch.stack(preds, dim=0)


def unnormalize(norm_frames, mean, std):
    return norm_frames * std[None, :, None, None] + mean[None, :, None, None]


def masked_mean(x, valid):
    denom = max(1, int(valid.sum()))
    return float(x[valid].sum() / denom)


def compute_frame_metrics(pred_phys, gt_phys, fluid_mask):
    # pred_phys, gt_phys: [T, C, H, W], mask: [H, W] bool where fluid is True.
    err = pred_phys - gt_phys
    abs_err = np.abs(err)
    sq_err = err * err

    vel_pred = np.sqrt(pred_phys[:, 0] ** 2 + pred_phys[:, 1] ** 2)
    vel_gt = np.sqrt(gt_phys[:, 0] ** 2 + gt_phys[:, 1] ** 2)
    vel_sq_err = (vel_pred - vel_gt) ** 2

    valid = fluid_mask.astype(np.float32)
    valid_sum = float(valid.sum())
    if valid_sum <= 0.0:
        raise ValueError("Fluid mask contains no valid cells.")

    valid_4d = valid[None, None, :, :]
    channel_count = float(pred_phys.shape[1])

    rmse_all = np.sqrt((sq_err * valid_4d).sum(axis=(1, 2, 3)) / (valid_sum * channel_count))
    mae_all = (abs_err * valid_4d).sum(axis=(1, 2, 3)) / (valid_sum * channel_count)
    rmse_u = np.sqrt((sq_err[:, 0] * valid).sum(axis=(1, 2)) / valid_sum)
    rmse_v = np.sqrt((sq_err[:, 1] * valid).sum(axis=(1, 2)) / valid_sum)
    rmse_p = np.sqrt((sq_err[:, 2] * valid).sum(axis=(1, 2)) / valid_sum)
    rmse_velmag = np.sqrt((vel_sq_err * valid).sum(axis=(1, 2)) / valid_sum)

    gt_sq = gt_phys * gt_phys
    num = np.sqrt((sq_err * valid_4d).sum(axis=(1, 2, 3)) / (valid_sum * channel_count))
    den = np.sqrt((gt_sq * valid_4d).sum(axis=(1, 2, 3)) / (valid_sum * channel_count)) + 1e-12
    rel_l2 = num / den

    # Per-pixel L2 error map used for visualization.
    pix_l2 = np.sqrt(np.sum(sq_err, axis=1))

    return {
        "rmse_all": np.asarray(rmse_all, dtype=np.float64),
        "mae_all": np.asarray(mae_all, dtype=np.float64),
        "rmse_u": np.asarray(rmse_u, dtype=np.float64),
        "rmse_v": np.asarray(rmse_v, dtype=np.float64),
        "rmse_p": np.asarray(rmse_p, dtype=np.float64),
        "rmse_velmag": np.asarray(rmse_velmag, dtype=np.float64),
        "rel_l2": np.asarray(rel_l2, dtype=np.float64),
        "pix_l2": pix_l2.astype(np.float32),
    }


def bootstrap_mean_ci(values, n_boot=500, alpha=0.05, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_metric(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def improvement_pct(base_vals, ft_vals, eps=1e-12):
    base_vals = np.asarray(base_vals, dtype=np.float64)
    ft_vals = np.asarray(ft_vals, dtype=np.float64)
    return 100.0 * np.mean((base_vals - ft_vals) / np.maximum(np.abs(base_vals), eps))


def write_metrics_csv(path, rows):
    fieldnames = [
        "sim",
        "frame_idx",
        "rollout_step",
        "base_rmse_all",
        "ft_rmse_all",
        "base_mae_all",
        "ft_mae_all",
        "base_rmse_u",
        "ft_rmse_u",
        "base_rmse_v",
        "ft_rmse_v",
        "base_rmse_p",
        "ft_rmse_p",
        "base_rmse_velmag",
        "ft_rmse_velmag",
        "base_rel_l2",
        "ft_rel_l2",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Also save a CSV that averages each metric per rollout step across the batch
    step_agg = {}
    numeric_fields = [
        "base_rmse_all",
        "ft_rmse_all",
        "base_mae_all",
        "ft_mae_all",
        "base_rmse_u",
        "ft_rmse_u",
        "base_rmse_v",
        "ft_rmse_v",
        "base_rmse_p",
        "ft_rmse_p",
        "base_rmse_velmag",
        "ft_rmse_velmag",
        "base_rel_l2",
        "ft_rel_l2",
    ]
    for r in rows:
        try:
            step = int(r.get("rollout_step", 0))
        except Exception:
            continue
        if step not in step_agg:
            step_agg[step] = {k: [] for k in numeric_fields}
        for k in numeric_fields:
            v = r.get(k)
            try:
                step_agg[step][k].append(float(v))
            except Exception:
                # skip non-numeric or missing
                pass

    run_dir = os.path.dirname(path) or "."
    stepwise_path = os.path.join(run_dir, "stepwise_mean_metrics.csv")
    with open(stepwise_path, "w", newline="") as fh:
        fieldnames_out = ["rollout_step", "n_samples"]
        for k in numeric_fields:
            fieldnames_out += [f"{k}_mean", f"{k}_std"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames_out)
        writer.writeheader()
        for step in sorted(step_agg.keys()):
            vals = step_agg[step]
            row = {"rollout_step": step, "n_samples": len(next(iter(vals.values()), []))}
            for k in numeric_fields:
                arr = np.array(vals[k], dtype=float) if len(vals[k]) > 0 else np.array([], dtype=float)
                if arr.size > 0:
                    row[f"{k}_mean"] = float(np.nanmean(arr))
                    row[f"{k}_std"] = float(np.nanstd(arr))
                else:
                    row[f"{k}_mean"] = ""
                    row[f"{k}_std"] = ""
            writer.writerow(row)


def plot_stepwise_rmse(path, base_step_rmse, ft_step_rmse):
    steps = np.arange(len(base_step_rmse))
    plt.figure(figsize=(8, 4.6), dpi=140)
    plt.plot(steps, base_step_rmse, label="Base checkpoint", linewidth=2)
    plt.plot(steps, ft_step_rmse, label="Fine-tuned checkpoint", linewidth=2)
    plt.xlabel("Rollout step")
    plt.ylabel("RMSE (all channels)")
    plt.title("Rollout Drift Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_error_maps(path, base_maps, ft_maps, mask_bool, frame_ids, sim_name, vmin, vmax):
    n = len(frame_ids)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), dpi=140)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="white")

    for j, step in enumerate(frame_ids):
        base_img = np.where(mask_bool, base_maps[step], np.nan)
        ft_img = np.where(mask_bool, ft_maps[step], np.nan)

        ax0 = axes[0, j]
        im0 = ax0.imshow(np.rot90(base_img, 1), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
        ax0.set_title(f"Base | step={step}", fontsize=18)
        ax0.axis("off")

        ax1 = axes[1, j]
        im1 = ax1.imshow(np.rot90(ft_img, 1), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
        ax1.set_title(f"Fine-tuned | step={step}", fontsize=18)
        ax1.axis("off")

    fig.suptitle(f"Per-pixel L2 Error Maps ({sim_name})\nMask=white, colormap=plasma, shared range", fontsize=11)
    fig.subplots_adjust(right=0.88, wspace=0.05, hspace=0.20)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Error magnitude", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(path)
    plt.close()


def _velocity_magnitude(frames):
    return np.sqrt(frames[:, 0] ** 2 + frames[:, 1] ** 2)


def _make_error_panel_arrays(gt_mag, base_mag, ft_mag, mask_bool):
    base_err = np.abs(base_mag - gt_mag)
    ft_err = np.abs(ft_mag - gt_mag)
    gt_mag = np.where(mask_bool[None, ...], gt_mag, np.nan)
    base_mag = np.where(mask_bool[None, ...], base_mag, np.nan)
    ft_mag = np.where(mask_bool[None, ...], ft_mag, np.nan)
    base_err = np.where(mask_bool[None, ...], base_err, np.nan)
    ft_err = np.where(mask_bool[None, ...], ft_err, np.nan)
    return gt_mag, base_mag, ft_mag, base_err, ft_err


def _comparison_axes(fig, state_vmax, err_vmax):
    fig.patch.set_facecolor("#111")
    # Leave more room for titles and colorbars by moving the right edge left
    grid = fig.add_gridspec(5, 1, left=0.05, right=0.80, top=0.94, bottom=0.05, hspace=0.28)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(5)]
    # Position colorbars inside the figure body (to the right of axes)
    state_cax = fig.add_axes([0.83, 0.56, 0.012, 0.26])
    error_cax = fig.add_axes([0.83, 0.16, 0.012, 0.26])

    state_cmap = plt.cm.viridis
    error_cmap = plt.cm.plasma.copy()
    error_cmap.set_bad(color="white")

    state_sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        cmap=state_cmap,
    )
    error_sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=err_vmax),
        cmap=error_cmap,
    )
    state_sm.set_array([])
    error_sm.set_array([])

    state_cbar = fig.colorbar(state_sm, cax=state_cax, orientation="vertical")
    error_cbar = fig.colorbar(error_sm, cax=error_cax, orientation="vertical")
    # Increase colorbar fonts for comparison visuals (doubled)
    state_cbar.set_label("|u|", color="white", fontsize=28)
    error_cbar.set_label("Absolute error", color="white", fontsize=28)
    state_cbar.ax.tick_params(colors="white", labelsize=24)
    error_cbar.ax.tick_params(colors="white", labelsize=24)
    for spine in state_cbar.ax.spines.values():
        spine.set_edgecolor("white")
    for spine in error_cbar.ax.spines.values():
        spine.set_edgecolor("white")

    for ax in axes:
        ax.set_facecolor("#111")
        ax.axis("off")

    return axes, state_cmap, error_cmap


def _draw_comparison_frame(axes, images, titles, cmaps, norms, frame_label=None):
    mappables = []
    for ax, image, title, cmap, norm in zip(axes, images, titles, cmaps, norms):
        im = ax.imshow(np.rot90(image, 1), origin="lower", aspect="equal", cmap=cmap, norm=norm, interpolation="bilinear")
        # Title font doubled for comparison snapshot/GIF
        ax.set_title(title, color="white", fontsize=18, fontweight="bold")
        mappables.append(im)
    return mappables


def _finite_max(*arrays, default=1.0):
    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(float(values.max()))
    if not finite_values:
        return float(default)
    vmax = max(finite_values)
    return float(vmax if np.isfinite(vmax) and vmax > 0.0 else default)


def save_comparison_snapshot(path, gt_phys, base_phys, ft_phys, mask_bool, frame_idx, sim_name):
    gt_mag = _velocity_magnitude(gt_phys)[frame_idx]
    base_mag = _velocity_magnitude(base_phys)[frame_idx]
    ft_mag = _velocity_magnitude(ft_phys)[frame_idx]
    gt_mag, base_mag, ft_mag, base_err, ft_err = _make_error_panel_arrays(
        gt_mag[None, ...], base_mag[None, ...], ft_mag[None, ...], mask_bool
    )
    gt_mag = gt_mag[0]
    base_mag = base_mag[0]
    ft_mag = ft_mag[0]
    base_err = base_err[0]
    ft_err = ft_err[0]

    state_vmax = _finite_max(gt_mag, base_mag, ft_mag)
    err_vmax = _finite_max(base_err, ft_err)

    # Widen figure to avoid clipping titles/labels
    fig = plt.figure(figsize=(14.8, 18.0), dpi=140)
    axes, cmap_state, cmap_err = _comparison_axes(fig, state_vmax, err_vmax)
    titles = ["GT |u|", "Pred_train |u|", "Pred_finetuned |u|", "Train error", "Fine-tuned error"]
    images = [gt_mag, base_mag, ft_mag, base_err, ft_err]
    cmaps = [cmap_state, cmap_state, cmap_state, cmap_err, cmap_err]
    norms = [
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=err_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=err_vmax),
    ]
    # Double the suptitle font for comparison snapshot and place slightly lower
    fig.suptitle(f"Velocity Magnitude Comparison | {sim_name} | frame {frame_idx}", color="white", fontsize=22, y=0.98)
    _draw_comparison_frame(axes, images, titles, cmaps, norms)

    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_comparison_gif(path, gt_phys, base_phys, ft_phys, mask_bool, sim_name):
    gt_mag = _velocity_magnitude(gt_phys)
    base_mag = _velocity_magnitude(base_phys)
    ft_mag = _velocity_magnitude(ft_phys)
    gt_mag, base_mag, ft_mag, base_err, ft_err = _make_error_panel_arrays(gt_mag, base_mag, ft_mag, mask_bool)

    state_vmax = _finite_max(gt_mag, base_mag, ft_mag)
    err_vmax = _finite_max(base_err, ft_err)

    writer = animation.PillowWriter(fps=10)
    # Widen GIF frames similarly to snapshot
    fig = plt.figure(figsize=(14.8, 18.0), dpi=120)
    axes, cmap_state, cmap_err = _comparison_axes(fig, state_vmax, err_vmax)
    fig.set_dpi(120)

    titles = ["GT |u|", "Pred_train |u|", "Pred_finetuned |u|", "Train error", "Fine-tuned error"]
    images = [gt_mag[0], base_mag[0], ft_mag[0], base_err[0], ft_err[0]]
    cmaps = [cmap_state, cmap_state, cmap_state, cmap_err, cmap_err]
    norms = [
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=state_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=err_vmax),
        matplotlib.colors.Normalize(vmin=0.0, vmax=err_vmax),
    ]
    artists = _draw_comparison_frame(axes, images, titles, cmaps, norms)

    def render_frame(frame_idx):
        # Double the suptitle font for GIF frames and keep it within the figure
        fig.suptitle(f"Velocity Magnitude Comparison | {sim_name} | frame {frame_idx:03d}/{gt_mag.shape[0]-1}", color="white", fontsize=22, y=0.98)
        frame_images = [gt_mag[frame_idx], base_mag[frame_idx], ft_mag[frame_idx], base_err[frame_idx], ft_err[frame_idx]]
        for artist, image in zip(artists, frame_images):
            artist.set_data(np.rot90(image, 1))

    with writer.saving(fig, path, dpi=120):
        for frame_idx in range(gt_mag.shape[0]):
            render_frame(frame_idx)
            writer.grab_frame(facecolor=fig.get_facecolor())

    plt.close(fig)


def write_markdown_report(path, info):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Rollout Error Comparison Report\n\n")
        f.write("## Setup\n")
        f.write(f"- Timestamp: {info['timestamp']}\n")
        f.write(f"- Simulation root: {info['sim_root']}\n")
        f.write(f"- Evaluated simulations: {info['n_eval_sims']}\n")
        f.write(f"- Total paired rollout frames: {info['n_paired_frames']}\n")
        f.write(f"- Base checkpoint: {info['base_ckpt']}\n")
        f.write(f"- Fine-tuned checkpoint: {info['ft_ckpt']}\n")
        f.write("\n")

        f.write("## Aggregate Metrics (Lower is better)\n")
        for metric_name in ["rmse_all", "mae_all", "rmse_velmag", "rmse_p", "rel_l2"]:
            b = info["base_summary"][metric_name]
            t = info["ft_summary"][metric_name]
            f.write(f"### {metric_name}\n")
            f.write(f"- Base mean: {b['mean']:.6f} (std={b['std']:.6f}, median={b['median']:.6f}, p90={b['p90']:.6f})\n")
            f.write(f"- Fine-tuned mean: {t['mean']:.6f} (std={t['std']:.6f}, median={t['median']:.6f}, p90={t['p90']:.6f})\n")
            f.write(f"- Mean relative improvement: {info['improvement_pct'][metric_name]:.3f}%\n")
            f.write("\n")

        f.write("## Paired Statistical Comparison (frame-wise, Fine-tuned - Base on rmse_all)\n")
        f.write(f"- Mean paired difference: {info['paired']['mean_diff']:.6f}\n")
        f.write(f"- Median paired difference: {info['paired']['median_diff']:.6f}\n")
        f.write(f"- 95% bootstrap CI of mean diff: [{info['paired']['ci_lo']:.6f}, {info['paired']['ci_hi']:.6f}]\n")
        f.write(f"- Win rate (fine-tuned better): {info['paired']['win_rate']:.2f}%\n")
        f.write(f"- Tie rate: {info['paired']['tie_rate']:.2f}%\n")
        f.write(f"- Loss rate (fine-tuned worse): {info['paired']['loss_rate']:.2f}%\n")
        f.write("\n")

        f.write("## Interpretation\n")
        if info["paired"]["mean_diff"] < 0 and info["paired"]["ci_hi"] < 0:
            interp = "Fine-tuned checkpoint shows a consistent rollout-error reduction versus base checkpoint."
        elif info["paired"]["mean_diff"] < 0:
            interp = "Fine-tuned checkpoint is better on average, but uncertainty overlaps zero for mean paired difference."
        else:
            interp = "Fine-tuned checkpoint does not improve mean rollout error under this evaluation protocol."
        f.write(f"- {interp}\n")
        f.write("- See CSV for per-frame detail and plots for drift/error-map behavior.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare rollout errors between base and fine-tuned checkpoints.")
    parser.add_argument("--sim-root", type=str, default=DEFAULT_SIM_ROOT)
    parser.add_argument("--base-ckpt", type=str, default=DEFAULT_BASE_CKPT)
    parser.add_argument("--finetuned-ckpt", type=str, default=DEFAULT_FINETUNED_CKPT)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--max-sims", type=int, default=0, help="0 means all evaluation simulations")
    parser.add_argument("--cache-workers", type=int, default=max(1, (os.cpu_count() or 4) - 3))
    parser.add_argument("--cache-states-filename", type=str, default="states.float32.npy")
    parser.add_argument("--cache-mask-filename", type=str, default="obstacle_mask.float32.npy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.base_ckpt):
        raise FileNotFoundError(f"Base checkpoint not found: {args.base_ckpt}")
    if not os.path.exists(args.finetuned_ckpt):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned_ckpt}")

    os.makedirs(args.out_dir, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_stamp)
    os.makedirs(run_dir, exist_ok=True)

    device = args.device
    use_amp = device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print(f"Device: {device}")
    print(f"Output directory: {run_dir}")
    print("Discovering simulations...")
    sim_infos = discover_simulations(args.sim_root)
    ensure_all_sim_caches(sim_infos, args.cache_workers, args.cache_states_filename, args.cache_mask_filename)

    train_sims, val_sims = split_simulations(sim_infos, args.val_frac)
    eval_sims = val_sims if val_sims else train_sims
    if args.max_sims and args.max_sims > 0:
        eval_sims = eval_sims[: args.max_sims]

    if not eval_sims:
        raise RuntimeError("No evaluation simulations available after split.")

    print(f"Train simulations: {len(train_sims)} | Eval simulations: {len(eval_sims)}")
    mean_np, std_np = compute_global_stats(train_sims, sim_infos)

    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    zero_norm = ((torch.zeros(3, device=device) - mean_t) / std_t).view(1, 3, 1, 1)
    labels = torch.tensor([1000], dtype=torch.long, device=device)

    print("Building/loading base model...")
    base_model = build_base_model(device)
    load_checkpoint_weights(base_model, args.base_ckpt, device, strict=True)
    base_model.eval()

    print("Building/loading fine-tuned model...")
    ft_model = build_finetuned_model(device)
    load_checkpoint_weights(ft_model, args.finetuned_ckpt, device, strict=True)
    ft_model.eval()

    rows = []
    agg = {
        "base": {"rmse_all": [], "mae_all": [], "rmse_u": [], "rmse_v": [], "rmse_p": [], "rmse_velmag": [], "rel_l2": []},
        "ft": {"rmse_all": [], "mae_all": [], "rmse_u": [], "rmse_v": [], "rmse_p": [], "rmse_velmag": [], "rel_l2": []},
    }
    step_buckets = {"base": {}, "ft": {}}

    vis_base_maps = None
    vis_ft_maps = None
    vis_mask = None
    vis_sim_name = None
    vis_gt_phys = None
    vis_base_phys = None
    vis_ft_phys = None

    for sim_idx, sim in enumerate(eval_sims):
        sim_name = os.path.basename(sim["dir"])
        states = np.asarray(load_packed_array(sim["states_path"]), dtype=np.float32)
        mask_np = np.asarray(load_packed_array(sim["packed_mask_path"]), dtype=np.float32)

        warmup = warmup_start_index(sim["n_frames"])
        usable_len = max(1, sim["n_frames"] - warmup)
        start_idx = warmup + int(usable_len * ROLLOUT_START_FRAC)
        end_idx = warmup + int(usable_len * ROLLOUT_END_FRAC)
        end_idx = min(sim["n_frames"], max(start_idx + 2, end_idx))
        rollout_len = end_idx - start_idx

        gt_phys = states[start_idx:end_idx]
        gt_norm = (gt_phys - mean_np[None, :, None, None]) / std_np[None, :, None, None]

        mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32).unsqueeze(0)
        first = torch.from_numpy(gt_norm[0]).to(device=device, dtype=torch.float32)

        base_norm = rollout_norm(base_model, first, mask_t, zero_norm, labels, rollout_len, use_amp)
        ft_norm = rollout_norm(ft_model, first, mask_t, zero_norm, labels, rollout_len, use_amp)

        base_phys = unnormalize(base_norm.detach().cpu().numpy(), mean_np, std_np)
        ft_phys = unnormalize(ft_norm.detach().cpu().numpy(), mean_np, std_np)

        fluid_mask = (mask_np[0] > 0.5)
        base_metrics = compute_frame_metrics(base_phys, gt_phys, fluid_mask)
        ft_metrics = compute_frame_metrics(ft_phys, gt_phys, fluid_mask)

        metric_keys = ["rmse_all", "mae_all", "rmse_u", "rmse_v", "rmse_p", "rmse_velmag", "rel_l2"]
        for key in metric_keys:
            agg["base"][key].extend(base_metrics[key].tolist())
            agg["ft"][key].extend(ft_metrics[key].tolist())

        for t in range(rollout_len):
            if t not in step_buckets["base"]:
                step_buckets["base"][t] = []
                step_buckets["ft"][t] = []
            step_buckets["base"][t].append(float(base_metrics["rmse_all"][t]))
            step_buckets["ft"][t].append(float(ft_metrics["rmse_all"][t]))

            rows.append(
                {
                    "sim": sim_name,
                    "frame_idx": int(start_idx + t),
                    "rollout_step": t,
                    "base_rmse_all": float(base_metrics["rmse_all"][t]),
                    "ft_rmse_all": float(ft_metrics["rmse_all"][t]),
                    "base_mae_all": float(base_metrics["mae_all"][t]),
                    "ft_mae_all": float(ft_metrics["mae_all"][t]),
                    "base_rmse_u": float(base_metrics["rmse_u"][t]),
                    "ft_rmse_u": float(ft_metrics["rmse_u"][t]),
                    "base_rmse_v": float(base_metrics["rmse_v"][t]),
                    "ft_rmse_v": float(ft_metrics["rmse_v"][t]),
                    "base_rmse_p": float(base_metrics["rmse_p"][t]),
                    "ft_rmse_p": float(ft_metrics["rmse_p"][t]),
                    "base_rmse_velmag": float(base_metrics["rmse_velmag"][t]),
                    "ft_rmse_velmag": float(ft_metrics["rmse_velmag"][t]),
                    "base_rel_l2": float(base_metrics["rel_l2"][t]),
                    "ft_rel_l2": float(ft_metrics["rel_l2"][t]),
                }
            )

        if vis_base_maps is None:
            vis_base_maps = base_metrics["pix_l2"]
            vis_ft_maps = ft_metrics["pix_l2"]
            vis_mask = fluid_mask
            vis_sim_name = sim_name
            vis_gt_phys = gt_phys
            vis_base_phys = base_phys
            vis_ft_phys = ft_phys

        print(f"Processed {sim_idx + 1}/{len(eval_sims)}: {sim_name} | rollout_len={rollout_len}")

    csv_path = os.path.join(run_dir, "paired_frame_metrics.csv")
    write_metrics_csv(csv_path, rows)

    step_keys = sorted(step_buckets["base"].keys())
    base_step_rmse = np.array([np.mean(step_buckets["base"][k]) for k in step_keys], dtype=np.float64)
    ft_step_rmse = np.array([np.mean(step_buckets["ft"][k]) for k in step_keys], dtype=np.float64)
    step_plot_path = os.path.join(run_dir, "rollout_drift_curve.png")
    plot_stepwise_rmse(step_plot_path, base_step_rmse, ft_step_rmse)

    global_map_max = float(max(np.max(vis_base_maps), np.max(vis_ft_maps))) if vis_base_maps is not None else 1.0
    map_path = os.path.join(run_dir, "error_maps_shared_scale.png")
    if vis_base_maps is not None:
        n_vis = min(6, vis_base_maps.shape[0])
        vis_ids = np.linspace(0, vis_base_maps.shape[0] - 1, n_vis, dtype=int).tolist()
        plot_error_maps(map_path, vis_base_maps, vis_ft_maps, vis_mask, vis_ids, vis_sim_name, 0.0, global_map_max)

    gif_path = os.path.join(run_dir, "comparison_rollout.gif")
    snapshot_path = os.path.join(run_dir, "comparison_snapshot.png")
    if vis_gt_phys is not None:
        last_idx = int(vis_gt_phys.shape[0] - 1)
        save_comparison_snapshot(snapshot_path, vis_gt_phys, vis_base_phys, vis_ft_phys, vis_mask, last_idx, vis_sim_name)
        save_comparison_gif(gif_path, vis_gt_phys, vis_base_phys, vis_ft_phys, vis_mask, vis_sim_name)

    base_summary = {k: summarize_metric(v) for k, v in agg["base"].items()}
    ft_summary = {k: summarize_metric(v) for k, v in agg["ft"].items()}

    improvement = {
        k: improvement_pct(np.asarray(agg["base"][k]), np.asarray(agg["ft"][k]))
        for k in ["rmse_all", "mae_all", "rmse_velmag", "rmse_p", "rel_l2"]
    }

    base_rmse_all = np.asarray(agg["base"]["rmse_all"], dtype=np.float64)
    ft_rmse_all = np.asarray(agg["ft"]["rmse_all"], dtype=np.float64)
    diffs = ft_rmse_all - base_rmse_all
    ci_lo, ci_hi = bootstrap_mean_ci(diffs)

    wins = np.sum(diffs < 0)
    ties = np.sum(np.isclose(diffs, 0.0, atol=1e-12))
    losses = np.sum(diffs > 0)
    total = max(1, diffs.size)

    report_info = {
        "timestamp": datetime.now().isoformat(),
        "sim_root": args.sim_root,
        "n_eval_sims": len(eval_sims),
        "n_paired_frames": int(diffs.size),
        "base_ckpt": args.base_ckpt,
        "ft_ckpt": args.finetuned_ckpt,
        "base_summary": base_summary,
        "ft_summary": ft_summary,
        "improvement_pct": improvement,
        "paired": {
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "win_rate": 100.0 * float(wins) / total,
            "tie_rate": 100.0 * float(ties) / total,
            "loss_rate": 100.0 * float(losses) / total,
        },
    }

    report_path = os.path.join(run_dir, "error_report.md")
    write_markdown_report(report_path, report_info)

    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Rollout error comparison complete.\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"CSV: {csv_path}\n")
        f.write(f"Drift plot: {step_plot_path}\n")
        f.write(f"Error maps: {map_path}\n")
        f.write(f"Snapshot: {snapshot_path}\n")
        f.write(f"GIF: {gif_path}\n")
        f.write(f"Report: {report_path}\n")
        f.write(f"Mean rmse_all (base): {base_summary['rmse_all']['mean']:.6f}\n")
        f.write(f"Mean rmse_all (fine-tuned): {ft_summary['rmse_all']['mean']:.6f}\n")
        f.write(f"Paired mean diff (ft-base): {report_info['paired']['mean_diff']:.6f}\n")
        f.write(f"95% CI: [{report_info['paired']['ci_lo']:.6f}, {report_info['paired']['ci_hi']:.6f}]\n")

    print("\nDone.")
    print(f"- CSV: {csv_path}")
    print(f"- Drift curve: {step_plot_path}")
    print(f"- Error maps: {map_path}")
    print(f"- Snapshot: {snapshot_path}")
    print(f"- GIF: {gif_path}")
    print(f"- Report: {report_path}")


if __name__ == "__main__":
    main()
