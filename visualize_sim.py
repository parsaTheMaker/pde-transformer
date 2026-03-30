"""
visualize_sim.py  –  fast multiprocess CFD visualiser
======================================================
  • Renders ONLY the last 80 % of frames (skips the warmup)
  • Color limits from 90th percentile → vivid, well-saturated colours
  • Each frame is rendered by a worker process in parallel
  • Frames are piped directly into ffmpeg → no temp files, no disk thrash

Output: /home/vatani/repos/pde-transformer/sim_000000_visualization.mp4
"""

import os, glob, subprocess, struct, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from io import BytesIO

# ─── Paths ────────────────────────────────────────────────────────────────────
SIM_DIR = "/home/vatani/repos/pde-transformer/data/sim_000000"
OUTPUT  = "/home/vatani/repos/pde-transformer/sim_000000_visualization.mp4"

# ─── Config ───────────────────────────────────────────────────────────────────
FPS        = 20     # output fps
SKIP_FRAC  = 0.20   # skip first 20 % (warmup)
PCTILE     = 100   # use full range (100th percentile)
DPI        = 130    # frame resolution
N_WORKERS  = max(1, cpu_count() - 1)   # parallel workers

# ─── File list – drop first 20 % ─────────────────────────────────────────────
vel_files = sorted(glob.glob(os.path.join(SIM_DIR, "velocity_*.npz")))
pre_files = sorted(glob.glob(os.path.join(SIM_DIR, "pressure_*.npz")))
assert len(vel_files) == len(pre_files)

start = int(len(vel_files) * SKIP_FRAC)
vel_files = vel_files[start:]
pre_files = pre_files[start:]
n_frames  = len(vel_files)
print(f"Total frames available: {len(vel_files)+start}  →  using last {n_frames} "
      f"(skipping first {start} = {SKIP_FRAC*100:.0f}%)")
print(f"Video duration: {n_frames/FPS:.1f} s at {FPS} fps")
print(f"Using {N_WORKERS} worker processes out of {cpu_count()} CPUs")

# ─── Obstacle mask ─────────────────────────────────────────────────────────────
obs_path  = os.path.join(SIM_DIR, "obstacle_mask.npz")
HAS_MASK  = os.path.exists(obs_path)
if HAS_MASK:
    obs_mask = np.load(obs_path)["arr_0"]  # int64: 1=fluid, 0=solid (cylinder)
    # The original karman.py saves: obsNp = obsNp[0] <= 0
    # so 1 = NOT obstacle (fluid), 0 = obstacle (cylinder) → paint cylinder only
    cylinder_mask = (obs_mask == 0)          # True where the cylinder is
    obs_plot = np.rot90(cylinder_mask, k=1)
    OBS_RGBA = np.zeros((*obs_plot.shape, 4), dtype=np.float32)
    OBS_RGBA[obs_plot] = [1.0, 1.0, 1.0, 1.0]   # solid white only on cylinder
else:
    OBS_RGBA = None
    print("Warning: obstacle_mask.npz not found")

# ─── Colour-limit scan (90th pctile) ──────────────────────────────────────────
print(f"Scanning {min(80, n_frames)} frames for full-range limits …")
sample_idx = np.linspace(0, n_frames - 1, min(80, n_frames), dtype=int)
vel_mags   = []
pres_vals  = []

for i in sample_idx:
    vel  = np.load(vel_files[i])["arr_0"]
    pres = np.load(pre_files[i])["arr_0"]
    mag  = np.sqrt(vel[0]**2 + vel[1]**2)
    vel_mags.append(mag)
    pres_vals.append(pres[0])

vel_vmax  = np.percentile(np.concatenate([m.ravel() for m in vel_mags]), PCTILE)
pres_flat = np.concatenate([p.ravel() for p in pres_vals])
pres_abs  = np.percentile(np.abs(pres_flat), PCTILE)

print(f"  |u| vmax  (full range): {vel_vmax:.4f}")
print(f"  |p| abs   (full range): {pres_abs:.5f}")

# ─── Build a small figure template once (shared state via module globals) ─────
FIG_W, FIG_H = 13, 4.2    # inches

def _make_fig():
    """Create the figure layout; return fig, ax_vel, ax_pres."""
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#111111")
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.05, right=0.95, top=0.86, bottom=0.18,
                            wspace=0.10)
    ax_vel  = fig.add_subplot(gs[0])
    ax_pres = fig.add_subplot(gs[1])
    for ax in (ax_vel, ax_pres):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#cccccc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.set_xlabel("x", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("y", color="#aaaaaa", fontsize=8)
    return fig, ax_vel, ax_pres


# ─── Worker: render one frame → raw RGBA bytes ────────────────────────────────
# All heavy imports happen in the worker process; state is passed via args.

def render_frame(args):
    idx, vf, pf, vel_vmax, pres_abs, obs_rgba, dt_per_frame, start, n_frames = args

    vel  = np.load(vf)["arr_0"]    # (2, x, y)
    pres = np.load(pf)["arr_0"]    # (1, x, y)
    mag  = np.rot90(np.sqrt(vel[0]**2 + vel[1]**2), k=1)
    p    = np.rot90(pres[0], k=1)

    fig, ax_vel, ax_pres = _make_fig()

    norm_vel  = Normalize(vmin=0,         vmax=vel_vmax)
    norm_pres = Normalize(vmin=-pres_abs, vmax=pres_abs)

    im_v = ax_vel.imshow(mag, origin="lower", aspect="auto",
                         cmap="viridis",  norm=norm_vel,  interpolation="bilinear")
    im_p = ax_pres.imshow(p,  origin="lower", aspect="auto",
                          cmap="RdBu_r",  norm=norm_pres, interpolation="bilinear")

    if obs_rgba is not None:
        ax_vel.imshow(obs_rgba,  origin="lower", aspect="auto", interpolation="nearest")
        ax_pres.imshow(obs_rgba, origin="lower", aspect="auto", interpolation="nearest")

    for ax, im, label, fmt in [
        (ax_vel,  im_v, "Velocity Magnitude  |u| (m/s)", "%.2f"),
        (ax_pres, im_p, "Pressure  p",                   "%.3f"),
    ]:
        ax.set_title(label, color="white", fontsize=11, pad=5, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                          pad=0.14, fraction=0.05, format=fmt)
        cb.ax.tick_params(colors="#cccccc", labelsize=7)
        cb.outline.set_edgecolor("#555555")

    # Absolute frame number and time
    abs_frame = start + idx
    t = abs_frame * dt_per_frame
    fig.text(0.5, 0.95,
             f"Frame {abs_frame:04d} / {start + n_frames - 1:04d}  |  t = {t:.3f} s",
             ha="center", va="top", color="white",
             fontsize=9, fontfamily="monospace")

    # Render to raw RGB bytes (no file I/O)
    buf = BytesIO()
    fig.savefig(buf, format="rgba", dpi=DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Frame dimensions (needed for ffmpeg pipe) ───────────────────────────────
# Render frame 0 once to get w×h
print("Probing frame size …")
_test = render_frame((0, vel_files[0], pre_files[0],
                      vel_vmax, pres_abs, OBS_RGBA,
                      0.05, start, n_frames))
# RGBA → each pixel = 4 bytes
_npx   = len(_test) // 4
_h_px  = int(round(FIG_H * DPI))
_w_px  = _npx // _h_px
# Re-derive exact dims from rendered buffer
_probe = np.frombuffer(_test, dtype=np.uint8).reshape(-1, 4)
_total_px = len(_probe)
# Use matplotlib to get exact figure size in pixels
_fig_tmp, _, _ = _make_fig()
_canvas = _fig_tmp.canvas
_canvas.draw()
_w_px, _h_px = _fig_tmp.canvas.get_width_height()
plt.close(_fig_tmp)
# Actually the easiest approach: just let ffmpeg figure it out via rawvideo + probing
# We'll store w,h from the first rendered frame properly:
_arr = np.frombuffer(_test, dtype=np.uint8)
# figure size in pixels = DPI * figsize_inches
_w_px_f = int(round(FIG_W * DPI))
_h_px_f = int(round(FIG_H * DPI))
# Make even (required by yuv420p)
_w_px_f += _w_px_f % 2
_h_px_f += _h_px_f % 2
print(f"Frame size: {_w_px_f} × {_h_px_f} px")

# ─── Open ffmpeg pipe ─────────────────────────────────────────────────────────
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-s", f"{_w_px_f}x{_h_px_f}",
    "-pix_fmt", "rgba",
    "-r", str(FPS),
    "-i", "pipe:0",
    "-vcodec", "libx264",
    "-crf", "18",
    "-preset", "fast",
    "-pix_fmt", "yuv420p",
    "-loglevel", "error",
    OUTPUT,
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# ─── Build argument list ──────────────────────────────────────────────────────
args_list = [
    (i, vel_files[i], pre_files[i], vel_vmax, pres_abs, OBS_RGBA, 0.05, start, n_frames)
    for i in range(n_frames)
]

# ─── Parallel render + stream to ffmpeg ──────────────────────────────────────
print(f"Rendering {n_frames} frames with {N_WORKERS} workers …")

CHUNK = N_WORKERS * 4   # process in ordered chunks so ffmpeg gets frames in order

with Pool(N_WORKERS) as pool:
    with tqdm(total=n_frames, unit="frame", desc="Encoding") as pbar:
        for chunk_start in range(0, n_frames, CHUNK):
            chunk_args = args_list[chunk_start : chunk_start + CHUNK]
            # imap preserves order, renders in parallel
            for raw_rgba in pool.imap(render_frame, chunk_args):
                ffmpeg_proc.stdin.write(raw_rgba)
            pbar.update(len(chunk_args))

ffmpeg_proc.stdin.close()
ret = ffmpeg_proc.wait()
if ret != 0:
    print(f"ERROR: ffmpeg exited with code {ret}", file=sys.stderr)
    sys.exit(ret)

print(f"\n✓ Video saved → {OUTPUT}")
