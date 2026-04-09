"""
MSE_vs_MIX.py
=============
frame-aligned comparison of two rollout-evaluation result folders.

This script reuses saved artifacts from two prior runs:
- rollout_step_metrics.csv
- mse_error_maps.npy
- grad_error_maps.npy
- direction_error_maps.npy

It creates:
1. A comparison GIF with GT velocity (viridis) + error maps (plasma) for both runs.
2. side-by-side metric plots on common frames.
3. Statistical summary CSV and TXT.
4. Aligned per-frame metrics CSV.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim_cache import load_packed_array


# -------------------------
# User-editable parameters.
# -------------------------
SIM_ROOT = "/home/vatani/data_vortex/256_inc"
TARGET_SIM_NAME = "sim_000082"

RUN_A_DIR = "runs/error_comparison/loss_evidence_MSE"
RUN_B_DIR = "runs/error_comparison/loss_evidence_MSE_GRAD_SIMILARITY"

RUN_A_LABEL = "Conventional"
RUN_B_LABEL = "Hybrid"

OUT_DIR = "runs/error_comparison/MSE_vs_MIX"

GIF_NAME = "comparison_gt_vel_vs_error_maps.gif"
GIF_FPS = 10
NUM_THREADS = max(1, (os.cpu_count() or 4) - 1)
GIF_SCALE = 2

TITLE_H = 34 * GIF_SCALE
TOP_MARGIN = 72 * GIF_SCALE
LEFT_MARGIN = 36 * GIF_SCALE
COL_GAP = 14 * GIF_SCALE
ROW_GAP = 56 * GIF_SCALE
BOTTOM_MARGIN = 28 * GIF_SCALE
OUTLINE_COLOR = "#b8b8b8"
TEXT_COLOR = "black"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LEGEND_BAR_H = 14 * GIF_SCALE
LEGEND_TEXT_SIZE = 14 * GIF_SCALE
LEGEND_LABEL_BAR_GAP = 8


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        return ImageFont.load_default()


def _read_step_csv(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {k: float(v) for k, v in row.items()}
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


@dataclass
class RunData:
    label: str
    rows: List[Dict[str, float]]
    pred_speed_maps: np.ndarray
    mse_maps: np.ndarray
    grad_maps: np.ndarray
    dir_maps: np.ndarray


@dataclass
class AlignedData:
    frames: np.ndarray
    a_rows: List[Dict[str, float]]
    b_rows: List[Dict[str, float]]
    a_speed: np.ndarray
    a_mse: np.ndarray
    a_grad: np.ndarray
    a_dir: np.ndarray
    b_speed: np.ndarray
    b_mse: np.ndarray
    b_grad: np.ndarray
    b_dir: np.ndarray


def _load_run(run_dir: str, label: str) -> RunData:
    step_csv = os.path.join(run_dir, "rollout_step_metrics.csv")
    speed_npy = os.path.join(run_dir, "predicted_speed_maps.npy")
    uv_npy = os.path.join(run_dir, "predicted_velocity_uv.npy")
    mse_npy = os.path.join(run_dir, "mse_error_maps.npy")
    grad_npy = os.path.join(run_dir, "grad_error_maps.npy")
    dir_npy = os.path.join(run_dir, "direction_error_maps.npy")

    for p in (step_csv, mse_npy, grad_npy, dir_npy):
        if not os.path.exists(p):
            raise RuntimeError(f"Missing required file: {p}")

    rows = _read_step_csv(step_csv)
    if os.path.exists(speed_npy):
        speed = np.load(speed_npy, mmap_mode="r")
    elif os.path.exists(uv_npy):
        uv = np.load(uv_npy, mmap_mode="r")
        if uv.ndim != 4 or uv.shape[1] < 2:
            raise RuntimeError(f"Unexpected predicted_velocity_uv.npy shape in {run_dir}: {uv.shape}")
        speed = np.sqrt(np.maximum(uv[:, 0] * uv[:, 0] + uv[:, 1] * uv[:, 1], 0.0)).astype(np.float32)
    else:
        raise RuntimeError(f"Missing both predicted_speed_maps.npy and predicted_velocity_uv.npy in {run_dir}")

    mse = np.load(mse_npy, mmap_mode="r")
    grad = np.load(grad_npy, mmap_mode="r")
    dire = np.load(dir_npy, mmap_mode="r")

    n = len(rows)
    if not (speed.shape[0] == n and mse.shape[0] == n and grad.shape[0] == n and dire.shape[0] == n):
        raise RuntimeError(
            f"Row/map length mismatch in {run_dir}: rows={n}, "
            f"speed={speed.shape[0]}, mse={mse.shape[0]}, grad={grad.shape[0]}, dir={dire.shape[0]}"
        )

    return RunData(label=label, rows=rows, pred_speed_maps=speed, mse_maps=mse, grad_maps=grad, dir_maps=dire)


def _align_runs(a: RunData, b: RunData) -> AlignedData:
    a_frames = np.array([int(r["global_frame"]) for r in a.rows], dtype=np.int32)
    b_frames = np.array([int(r["global_frame"]) for r in b.rows], dtype=np.int32)

    common_frames, a_idx, b_idx = np.intersect1d(a_frames, b_frames, assume_unique=True, return_indices=True)
    if common_frames.size == 0:
        raise RuntimeError("No overlapping global frames between the two runs.")

    a_rows = [a.rows[int(i)] for i in a_idx]
    b_rows = [b.rows[int(i)] for i in b_idx]

    return AlignedData(
        frames=common_frames.astype(np.int32, copy=False),
        a_rows=a_rows,
        b_rows=b_rows,
        a_speed=a.pred_speed_maps[a_idx],
        a_mse=a.mse_maps[a_idx],
        a_grad=a.grad_maps[a_idx],
        a_dir=a.dir_maps[a_idx],
        b_speed=b.pred_speed_maps[b_idx],
        b_mse=b.mse_maps[b_idx],
        b_grad=b.grad_maps[b_idx],
        b_dir=b.dir_maps[b_idx],
    )


def _load_target_states_and_mask(sim_root: str, target_sim_name: str):
    target_dir = os.path.join(sim_root, target_sim_name)
    if not os.path.isdir(target_dir):
        raise RuntimeError(f"Simulation {target_sim_name} not found under {sim_root}")

    states_path = os.path.join(target_dir, "states.float32.npy")
    mask_path = os.path.join(target_dir, "obstacle_mask.float32.npy")
    if not os.path.exists(states_path):
        raise RuntimeError(f"Missing cached states file: {states_path}")
    if not os.path.exists(mask_path):
        raise RuntimeError(f"Missing cached mask file: {mask_path}")

    states = load_packed_array(states_path)
    mask = load_packed_array(mask_path)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape: {mask.shape}")

    return states, mask


def _gt_velocity_maps(states: np.ndarray, frames: np.ndarray) -> np.ndarray:
    max_frame = states.shape[0] - 1
    if frames.min(initial=0) < 0 or frames.max(initial=0) > max_frame:
        raise RuntimeError(f"Frame range [{frames.min()}, {frames.max()}] out of bounds for GT states length {states.shape[0]}")

    selected = states[frames, :2]  # [T, 2, H, W]
    vel_sq = np.multiply(selected[:, 0], selected[:, 0], dtype=np.float32)
    vel_sq += np.multiply(selected[:, 1], selected[:, 1], dtype=np.float32)
    np.maximum(vel_sq, 0.0, out=vel_sq)
    np.sqrt(vel_sq, out=vel_sq)
    return vel_sq


def _rotate_and_mask_stack_fast(stack: np.ndarray, mask_rot: np.ndarray) -> np.ma.MaskedArray:
    # Precompute once so animation update only swaps frame references.
    rot = np.rot90(stack, k=3, axes=(1, 2))
    mask_3d = np.broadcast_to(mask_rot[None, :, :], rot.shape)
    return np.ma.MaskedArray(rot, mask=~mask_3d, copy=False)


def _frame_to_rgba(frame: np.ma.MaskedArray, cmap, vmin: float, vmax: float) -> Image.Image:
    span = vmax - vmin
    if span <= 0.0:
        normed = np.zeros(frame.shape, dtype=np.float32)
    else:
        normed = (frame - vmin) / span
    rgba = cmap(np.ma.masked_invalid(normed), bytes=True)
    return Image.fromarray(rgba, mode="RGBA")


def _make_horizontal_legend(cmap, vmin: float, vmax: float, width: int, font) -> Image.Image:
    legend_w = max(64, width)
    label_font = _load_font(LEGEND_TEXT_SIZE)
    label_h = label_font.getbbox("0")[3] - label_font.getbbox("0")[1]
    legend_h = label_h + LEGEND_LABEL_BAR_GAP + LEGEND_BAR_H
    legend = Image.new("RGBA", (legend_w, legend_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(legend)

    grad = np.linspace(0.0, 1.0, legend_w, dtype=np.float32)[None, :]
    rgba = cmap(grad, bytes=True)
    grad_img = Image.fromarray(rgba, mode="RGBA").resize((legend_w, LEGEND_BAR_H), resample=Image.Resampling.NEAREST)
    legend.paste(grad_img, (0, label_h + LEGEND_LABEL_BAR_GAP))

    left_text = f"{vmin:.2g}"
    right_text = f"{vmax:.2g}"
    left_box = draw.textbbox((0, 0), left_text, font=label_font)
    right_box = draw.textbbox((0, 0), right_text, font=label_font)
    text_y = 0
    draw.text((0, text_y), left_text, fill=TEXT_COLOR, font=label_font)
    draw.text((legend_w - (right_box[2] - right_box[0]), text_y), right_text, fill=TEXT_COLOR, font=label_font)
    draw.rectangle(
        [0, label_h + LEGEND_LABEL_BAR_GAP, legend_w - 1, label_h + LEGEND_LABEL_BAR_GAP + LEGEND_BAR_H - 1],
        outline="#666666",
        width=1,
    )
    return legend


def _compose_gif_frame(
    i: int,
    gt_stack: np.ma.MaskedArray,
    a_speed_stack: np.ma.MaskedArray,
    a_mse_stack: np.ma.MaskedArray,
    a_grad_stack: np.ma.MaskedArray,
    a_dir_stack: np.ma.MaskedArray,
    b_speed_stack: np.ma.MaskedArray,
    b_mse_stack: np.ma.MaskedArray,
    b_grad_stack: np.ma.MaskedArray,
    b_dir_stack: np.ma.MaskedArray,
    frames: np.ndarray,
    a_label: str,
    b_label: str,
    cmap_vel,
    cmap_err,
    vmins: dict,
    vmaxs: dict,
    fonts: dict,
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    canvas_w = LEFT_MARGIN + (5 * panel_w) + (4 * COL_GAP) + 18
    canvas_h = TOP_MARGIN + (2 * panel_h) + (2 * TITLE_H) + ROW_GAP + BOTTOM_MARGIN
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    title_font = fonts["title"]
    frame_font = fonts["frame"]

    draw.text((18 * GIF_SCALE, 16 * GIF_SCALE), f"GT + Error Comparison | step={i + 1:03d}/{gt_stack.shape[0]:03d}", fill=TEXT_COLOR, font=frame_font)

    title_grid = [
        ["GT Velocity", f"{a_label}: Pred Velocity", f"{a_label}: MSE Error", f"{a_label}: Grad Error", f"{a_label}: Dir Error"],
        ["GT Velocity", f"{b_label}: Pred Velocity", f"{b_label}: MSE Error", f"{b_label}: Grad Error", f"{b_label}: Dir Error"],
    ]

    stacks = [
        [gt_stack, a_speed_stack, a_mse_stack, a_grad_stack, a_dir_stack],
        [gt_stack, b_speed_stack, b_mse_stack, b_grad_stack, b_dir_stack],
    ]
    cmaps = [
        [cmap_vel, cmap_vel, cmap_err, cmap_err, cmap_err],
        [cmap_vel, cmap_vel, cmap_err, cmap_err, cmap_err],
    ]
    keys = [
        ["vel", "pred_vel", "mse", "grad", "dir"],
        ["vel", "pred_vel", "mse", "grad", "dir"],
    ]

    legend_cmaps = [cmap_vel, cmap_vel, cmap_err, cmap_err, cmap_err]
    legend_keys = ["vel", "pred_vel", "mse", "grad", "dir"]
    legend_w = max(96 * GIF_SCALE, panel_w - 28 * GIF_SCALE)
    legends = [
        _make_horizontal_legend(legend_cmaps[col_idx], vmins[legend_keys[col_idx]], vmaxs[legend_keys[col_idx]], legend_w, frame_font)
        for col_idx in range(5)
    ]

    for row_idx in range(2):
        row_y = TOP_MARGIN + row_idx * (panel_h + TITLE_H + ROW_GAP)
        for col_idx in range(5):
            cell_x = LEFT_MARGIN + col_idx * (panel_w + COL_GAP)
            title = title_grid[row_idx][col_idx]
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_w = title_bbox[2] - title_bbox[0]
            draw.text((cell_x + (panel_w - title_w) / 2, row_y), title, fill=TEXT_COLOR, font=title_font)

            frame = stacks[row_idx][col_idx][i]
            panel = _frame_to_rgba(frame, cmaps[row_idx][col_idx], vmins[keys[row_idx][col_idx]], vmaxs[keys[row_idx][col_idx]])
            panel = panel.resize((panel_w, panel_h), resample=Image.Resampling.LANCZOS)
            canvas.paste(panel, (cell_x, row_y + TITLE_H))
            draw.rectangle(
                [cell_x, row_y + TITLE_H, cell_x + panel_w - 1, row_y + TITLE_H + panel_h - 1],
                outline=OUTLINE_COLOR,
                width=1,
            )

    legend_y = TOP_MARGIN + panel_h + TITLE_H + 10
    legend_x_pad = max(2, (panel_w - legend_w) // 2)
    for col_idx, legend in enumerate(legends):
        cell_x = LEFT_MARGIN + col_idx * (panel_w + COL_GAP)
        canvas.paste(legend, (cell_x + legend_x_pad, legend_y), legend)

    return canvas.convert("P", palette=Image.ADAPTIVE, colors=256)


def _save_aligned_csv(aligned: AlignedData, out_csv: str):
    keys = [
        "frame",
        "step_index",
        "a_mse",
        "b_mse",
        "a_grad",
        "b_grad",
        "a_direction",
        "b_direction",
        "delta_mse",
        "delta_grad",
        "delta_direction",
        "gain_mse_pct",
        "gain_grad_pct",
        "gain_direction_pct",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i, frame in enumerate(aligned.frames):
            a_m = aligned.a_rows[i]["mse_mean"]
            b_m = aligned.b_rows[i]["mse_mean"]
            a_g = aligned.a_rows[i]["grad_mean"]
            b_g = aligned.b_rows[i]["grad_mean"]
            a_d = aligned.a_rows[i]["direction_mean"]
            b_d = aligned.b_rows[i]["direction_mean"]

            row = {
                "frame": int(frame),
                "step_index": i + 1,
                "a_mse": a_m,
                "b_mse": b_m,
                "a_grad": a_g,
                "b_grad": b_g,
                "a_direction": a_d,
                "b_direction": b_d,
                "delta_mse": b_m - a_m,
                "delta_grad": b_g - a_g,
                "delta_direction": b_d - a_d,
                "gain_mse_pct": 100.0 * (a_m - b_m) / (a_m + 1e-12),
                "gain_grad_pct": 100.0 * (a_g - b_g) / (a_g + 1e-12),
                "gain_direction_pct": 100.0 * (a_d - b_d) / (a_d + 1e-12),
            }
            w.writerow(row)


def _save_summary(aligned: AlignedData, a_label: str, b_label: str, out_csv: str, out_txt: str):
    def arr(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    a_m = arr(aligned.a_rows, "mse_mean")
    b_m = arr(aligned.b_rows, "mse_mean")
    a_g = arr(aligned.a_rows, "grad_mean")
    b_g = arr(aligned.b_rows, "grad_mean")
    a_d = arr(aligned.a_rows, "direction_mean")
    b_d = arr(aligned.b_rows, "direction_mean")

    def metric_summary(name: str, x: np.ndarray, y: np.ndarray):
        gain = 100.0 * (x - y) / (x + 1e-12)
        return {
            "metric": name,
            f"mean_{a_label}": float(np.mean(x)),
            f"mean_{b_label}": float(np.mean(y)),
            "mean_delta_b_minus_a": float(np.mean(y - x)),
            "mean_gain_pct_a_to_b": float(np.mean(gain)),
            "median_gain_pct_a_to_b": float(np.median(gain)),
            "p95_gain_pct_a_to_b": float(np.percentile(gain, 95)),
            "win_rate_b_better_pct": float(100.0 * np.mean(y < x)),
        }

    summaries = [
        metric_summary("mse_mean", a_m, b_m),
        metric_summary("grad_mean", a_g, b_g),
        metric_summary("direction_mean", a_d, b_d),
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Comparison Study\n")
        f.write("=" * 80 + "\n")
        f.write(f"Frames compared: {len(aligned.frames)}\n")
        f.write(f"Run A: {a_label}\n")
        f.write(f"Run B: {b_label}\n\n")
        for s in summaries:
            f.write(
                f"- {s['metric']}: mean({a_label})={s[f'mean_{a_label}']:.6e}, "
                f"mean({b_label})={s[f'mean_{b_label}']:.6e}, "
                f"delta(B-A)={s['mean_delta_b_minus_a']:.6e}, "
                f"mean_gain_pct(A->B)={s['mean_gain_pct_a_to_b']:.3f}%, "
                f"win_rate(B better)={s['win_rate_b_better_pct']:.2f}%\n"
            )


def _plot_comparison(aligned: AlignedData, a_label: str, b_label: str, out_png: str):
    step = np.arange(1, len(aligned.frames) + 1, dtype=np.float64)

    def arr(rows, key):
        return np.array([r[key] for r in rows], dtype=np.float64)

    a_m = arr(aligned.a_rows, "mse_mean")
    b_m = arr(aligned.b_rows, "mse_mean")
    a_g = arr(aligned.a_rows, "grad_mean")
    b_g = arr(aligned.b_rows, "grad_mean")
    a_d = arr(aligned.a_rows, "direction_mean")
    b_d = arr(aligned.b_rows, "direction_mean")

    gain_m = 100.0 * (a_m - b_m) / (a_m + 1e-12)
    gain_g = 100.0 * (a_g - b_g) / (a_g + 1e-12)
    gain_d = 100.0 * (a_d - b_d) / (a_d + 1e-12)

    d2_a = np.gradient(np.gradient(a_m))
    d2_b = np.gradient(np.gradient(b_m))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=200, facecolor="white")
    for ax in axes.ravel():
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor("#777")

    ax = axes[0, 0]
    color_m = "#1f77b4"
    color_g = "#ff7f0e"
    color_d = "#2ca02c"
    ax.plot(step, a_m, color=color_m, lw=2, ls="--", label=f"{a_label} MSE")
    ax.plot(step, b_m, color=color_m, lw=2, ls="-", label=f"{b_label} MSE")
    ax.plot(step, a_g, color=color_g, lw=1.8, ls="--", label=f"{a_label} Grad")
    ax.plot(step, b_g, color=color_g, lw=1.8, ls="-", label=f"{b_label} Grad")
    ax.plot(step, a_d, color=color_d, lw=1.8, ls="--", label=f"{a_label} Dir")
    ax.plot(step, b_d, color=color_d, lw=1.8, ls="-", label=f"{b_label} Dir")
    ax.set_yscale("log")
    ax.set_title("Error Curves (Aligned Frames)", fontsize=14, color="black")
    ax.set_xlabel("Aligned step", fontsize=12, color="black")
    ax.set_ylabel("Error (log)", fontsize=12, color="black")
    ax.legend(fontsize=9, framealpha=0.2, ncol=2)

    ax = axes[0, 1]
    ax.plot(step, gain_m, color="#d62728", lw=2, label="MSE gain %")
    ax.plot(step, gain_g, color="#ff7f0e", lw=2, label="Grad gain %")
    ax.plot(step, gain_d, color="#9467bd", lw=2, label="Direction gain %")
    ax.axhline(0.0, color="#777", lw=1.0, ls="--")
    ax.set_title("Relative Gain from A to B", fontsize=14, color="black")
    ax.set_xlabel("Aligned step", fontsize=12, color="black")
    ax.set_ylabel("Gain (%)", fontsize=12, color="black")
    ax.legend(fontsize=10, framealpha=0.2)

    ax = axes[1, 0]
    ax.plot(step, a_m, color="#1f77b4", lw=2, label=f"{a_label} MSE")
    ax.plot(step, b_m, color="#d62728", lw=2, label=f"{b_label} MSE")
    ax.set_title("MSE-only Curve Comparison", fontsize=14, color="black")
    ax.set_xlabel("Aligned step", fontsize=12, color="black")
    ax.set_ylabel("MSE", fontsize=12, color="black")
    ax.legend(fontsize=10, framealpha=0.2)

    ax = axes[1, 1]
    ax.plot(step, d2_a, color="#1f77b4", lw=2, label=f"{a_label} d2(MSE)")
    ax.plot(step, d2_b, color="#d62728", lw=2, label=f"{b_label} d2(MSE)")
    ax.axhline(0.0, color="#777", lw=1.0, ls="--")
    ax.set_title("Second Derivative of MSE", fontsize=14, color="black")
    ax.set_xlabel("Aligned step", fontsize=12, color="black")
    ax.set_ylabel("d2(MSE)/d(step)^2", fontsize=12, color="black")
    ax.legend(fontsize=10, framealpha=0.2)

    plt.tight_layout()
    plt.savefig(out_png, facecolor="white")
    plt.close(fig)


def _make_comparison_gif(
    gt_vel: np.ndarray,
    a_speed: np.ndarray,
    a_mse: np.ndarray,
    a_grad: np.ndarray,
    a_dir: np.ndarray,
    b_speed: np.ndarray,
    b_mse: np.ndarray,
    b_grad: np.ndarray,
    b_dir: np.ndarray,
    mask: np.ndarray,
    frames: np.ndarray,
    a_label: str,
    b_label: str,
    out_gif: str,
):
    n = gt_vel.shape[0]

    # Shared scales for visual comparison.
    vmax_vel = float(np.percentile(gt_vel, 99.0)) + 1e-12
    vmax_pred_vel = float(np.percentile(np.concatenate([a_speed, b_speed], axis=0), 99.0)) + 1e-12
    vmax_mse = float(np.percentile(np.concatenate([a_mse, b_mse], axis=0), 99.0)) + 1e-12
    vmax_grad = float(np.percentile(np.concatenate([a_grad, b_grad], axis=0), 99.0)) + 1e-12
    vmax_dir = float(np.percentile(np.concatenate([a_dir, b_dir], axis=0), 99.0)) + 1e-12

    m = np.rot90(mask > 0.5, k=3)
    gt_stack = _rotate_and_mask_stack_fast(gt_vel, m)
    a_speed_stack = _rotate_and_mask_stack_fast(a_speed, m)
    a_mse_stack = _rotate_and_mask_stack_fast(a_mse, m)
    a_grad_stack = _rotate_and_mask_stack_fast(a_grad, m)
    a_dir_stack = _rotate_and_mask_stack_fast(a_dir, m)
    b_speed_stack = _rotate_and_mask_stack_fast(b_speed, m)
    b_mse_stack = _rotate_and_mask_stack_fast(b_mse, m)
    b_grad_stack = _rotate_and_mask_stack_fast(b_grad, m)
    b_dir_stack = _rotate_and_mask_stack_fast(b_dir, m)

    cmap_vel = plt.cm.viridis.copy()
    cmap_vel.set_bad(color="white")
    cmap_err = plt.cm.plasma.copy()
    cmap_err.set_bad(color="white")
    fonts = {
        "title": _load_font(20 * GIF_SCALE),
        "frame": _load_font(22 * GIF_SCALE),
    }
    vmins = {
        "vel": 0.0,
        "pred_vel": 0.0,
        "mse": 0.0,
        "grad": 0.0,
        "dir": 0.0,
    }
    vmaxs = {
        "vel": vmax_vel,
        "pred_vel": vmax_pred_vel,
        "mse": vmax_mse,
        "grad": vmax_grad,
        "dir": vmax_dir,
    }
    panel_h, panel_w = int(gt_stack.shape[1] * GIF_SCALE), int(gt_stack.shape[2] * GIF_SCALE)

    first_frame = _compose_gif_frame(
        0,
        gt_stack,
        a_speed_stack,
        a_mse_stack,
        a_grad_stack,
        a_dir_stack,
        b_speed_stack,
        b_mse_stack,
        b_grad_stack,
        b_dir_stack,
        frames,
        a_label,
        b_label,
        cmap_vel,
        cmap_err,
        vmins,
        vmaxs,
        fonts,
        panel_w,
        panel_h,
    )

    def frame_iter():
        for i in range(1, n):
            yield _compose_gif_frame(
                i,
                gt_stack,
                a_speed_stack,
                a_mse_stack,
                a_grad_stack,
                a_dir_stack,
                b_speed_stack,
                b_mse_stack,
                b_grad_stack,
                b_dir_stack,
                frames,
                a_label,
                b_label,
                cmap_vel,
                cmap_err,
                vmins,
                vmaxs,
                fonts,
                panel_w,
                panel_h,
            )

    first_frame.save(
        out_gif,
        save_all=True,
        append_images=frame_iter(),
        duration=int(round(1000.0 / GIF_FPS)),
        loop=0,
        optimize=False,
        disposal=2,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with ThreadPoolExecutor(max_workers=min(2, NUM_THREADS)) as pool:
        fut_a = pool.submit(_load_run, RUN_A_DIR, RUN_A_LABEL)
        fut_b = pool.submit(_load_run, RUN_B_DIR, RUN_B_LABEL)
        run_a = fut_a.result()
        run_b = fut_b.result()

    aligned = _align_runs(run_a, run_b)
    states, mask = _load_target_states_and_mask(SIM_ROOT, TARGET_SIM_NAME)
    gt_vel = _gt_velocity_maps(states, aligned.frames)

    aligned_csv = os.path.join(OUT_DIR, "aligned_frame_metrics.csv")
    summary_csv = os.path.join(OUT_DIR, "comparison_summary.csv")
    summary_txt = os.path.join(OUT_DIR, "comparison_summary.txt")
    plot_png = os.path.join(OUT_DIR, "comparison_plots.png")
    gif_path = os.path.join(OUT_DIR, GIF_NAME)

    _save_aligned_csv(aligned, aligned_csv)
    _save_summary(aligned, run_a.label, run_b.label, summary_csv, summary_txt)
    _plot_comparison(aligned, run_a.label, run_b.label, plot_png)
    _make_comparison_gif(
        gt_vel=gt_vel,
        a_speed=aligned.a_speed,
        a_mse=aligned.a_mse,
        a_grad=aligned.a_grad,
        a_dir=aligned.a_dir,
        b_speed=aligned.b_speed,
        b_mse=aligned.b_mse,
        b_grad=aligned.b_grad,
        b_dir=aligned.b_dir,
        mask=mask,
        frames=aligned.frames,
        a_label=run_a.label,
        b_label=run_b.label,
        out_gif=gif_path,
    )

    print("Artifacts saved:")
    print(f"- Aligned CSV    : {aligned_csv}")
    print(f"- Summary CSV    : {summary_csv}")
    print(f"- Summary TXT    : {summary_txt}")
    print(f"- Comparison PNG : {plot_png}")
    print(f"- Comparison GIF : {gif_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
