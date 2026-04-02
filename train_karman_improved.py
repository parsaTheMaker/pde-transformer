"""
train_karman.py
===============
Train PDE-Transformer (MC-S) on the Karman vortex CFD simulations under SIM_ROOT.

Each sim_* folder is treated as one independent simulation. Frames are never mixed
across simulations for train/validation splits or for visualization.
"""

import math
import os
import sys
from contextlib import suppress
from datetime import datetime
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
from sim_cache import discover_simulations, ensure_all_sim_caches, load_npz_array, load_packed_array


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
SIM_ROOT = "./256_inc"  # e.g. r"D:\data\256_inc" or "/data/256_inc"
OUT_DIR = os.path.join("runs", "karman")
EPOCHS = 40
BATCH_SIZE = 28
ACCUM_GRAD = 1
LR = 4e-5
VAL_FRAC = 0.10
WARMUP_FRAC = 0.3
MSE_LOSS_WEIGHT = 1.0
SPEC_LOSS_WEIGHT = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS_VID = 10
VID_FRAMES = 50
DPI_VID = 110
SKIP_TRAIN = False
RESUME_CHECKPOINT = "./runs/karman/last.ckpt"  # e.g. r"D:\runs\karman\last.ckpt" or "/home/user/last.ckpt"
MODEL_TYPE = "PDE-S"  # Smallest PDETransformer variant in this repo
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
# This model is attention-heavy and does many explicit permutes/window reshapes,
# so channels_last is not a reliable speedup here.
USE_CHANNELS_LAST = False
NUM_WORKERS = max(0, cpu_count() - 3)
PIN_MEMORY = DEVICE == "cuda"
CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
CACHE_WORKERS = max(1, cpu_count() - 3)
PREFETCH_FACTOR = 6
TQDM_UPDATE_EVERY = 20

def packed_slice_to_numpy(array):
    return np.array(array, dtype=np.float32, copy=True)


_HANN_WINDOW_CACHE = {}


def get_hann_window2d(height, width, device, dtype):
    key = (height, width, str(device), dtype)
    window2d = _HANN_WINDOW_CACHE.get(key)
    if window2d is None:
        wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
        wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
        window2d = (wy[:, None] * wx[None, :]).unsqueeze(0).unsqueeze(0)
        _HANN_WINDOW_CACHE[key] = window2d
    return window2d

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
        sample_count = max(1, int(round(target_samples * (usable_frames / total_frames))))
        idxs = np.linspace(start_idx, sim["n_frames"] - 1, sample_count, dtype=int)
        samples.append(np.asarray(states[idxs], dtype=np.float32))

    if not samples:
        sim = source_sims[0]
        states = load_packed_array(sim["states_path"])
        samples.append(np.asarray(states[[0]], dtype=np.float32))

    stacked = np.concatenate(samples, axis=0)
    mean = torch.tensor(stacked.mean(axis=(0, 2, 3)), dtype=torch.float32)
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + 1e-6
    return mean, std


class MultiSimKarmanDataset(Dataset):
    def __init__(self, sim_list, mean, std):
        self.sims = sim_list
        self.mean_np = mean[:, None, None].numpy().astype(np.float32, copy=True)
        self.inv_std_np = (1.0 / std[:, None, None].numpy()).astype(np.float32, copy=True)
        self.samples = []
        self._state_arrays = {}
        self._mask_tensors = {}

        for sim_idx, sim in enumerate(self.sims):
            start_idx = warmup_start_index(sim["n_frames"])
            for frame_idx in range(start_idx, sim["n_frames"] - 1):
                self.samples.append((sim_idx, frame_idx))

    def __len__(self):
        return len(self.samples)

    def _get_states(self, sim_idx):
        states = self._state_arrays.get(sim_idx)
        if states is None:
            states = load_packed_array(self.sims[sim_idx]["states_path"])
            self._state_arrays[sim_idx] = states
        return states

    def _get_mask(self, sim_idx):
        mask_t = self._mask_tensors.get(sim_idx)
        if mask_t is None:
            mask_np = packed_slice_to_numpy(load_packed_array(self.sims[sim_idx]["packed_mask_path"]))
            mask_t = torch.from_numpy(mask_np).float()
            self._mask_tensors[sim_idx] = mask_t
        return mask_t

    def __getitem__(self, idx):
        sim_idx, frame_idx = self.samples[idx]
        states = self._get_states(sim_idx)
        x_np = packed_slice_to_numpy(states[frame_idx])
        y_np = packed_slice_to_numpy(states[frame_idx + 1])
        np.subtract(x_np, self.mean_np, out=x_np)
        np.subtract(y_np, self.mean_np, out=y_np)
        np.multiply(x_np, self.inv_std_np, out=x_np)
        np.multiply(y_np, self.inv_std_np, out=y_np)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x.float(), y.float(), self._get_mask(sim_idx)


def hybrid_spatial_spectral_loss(pred, target, eps=1e-6):
    spatial_mse = F.mse_loss(pred, target)
    height, width = pred.shape[-2:]
    window2d = get_hann_window2d(height, width, pred.device, pred.dtype)

    pred_win = pred * window2d
    target_win = target * window2d
    pred_fft = torch.fft.rfft2(pred_win, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target_win, dim=(-2, -1), norm="ortho")
    diff_power = (pred_fft - target_fft).abs().pow(2)
    target_power = target_fft.abs().pow(2)
    spectral_loss = (diff_power / (target_power.detach() + eps)).mean()

    total_loss = MSE_LOSS_WEIGHT * spatial_mse + SPEC_LOSS_WEIGHT * spectral_loss
    return total_loss, spatial_mse.detach(), spectral_loss.detach()


def build_loader(dataset, shuffle):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    if PIN_MEMORY:
        kwargs["pin_memory_device"] = DEVICE
    return DataLoader(dataset, **kwargs)


class DevicePrefetchLoader:
    def __init__(self, loader, device, use_channels_last):
        self.loader = loader
        self.device = device
        self.use_channels_last = use_channels_last
        self.enabled = device == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=device) if self.enabled else None

    def __len__(self):
        return len(self.loader)

    def _move_batch(self, batch):
        x, y, mask = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
            mask = mask.contiguous(memory_format=torch.channels_last)
        return x, y, mask

    def __iter__(self):
        if not self.enabled:
            for batch in self.loader:
                yield batch
            return

        loader_iter = iter(self.loader)
        next_batch = None

        def preload():
            nonlocal next_batch
            try:
                batch = next(loader_iter)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(self.stream):
                next_batch = self._move_batch(batch)

        preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = next_batch
            preload()
            yield batch


def make_progress(iterable, desc):
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )


def maybe_wrap_prefetch(loader):
    if loader is None or DEVICE != "cuda":
        return loader
    return DevicePrefetchLoader(loader, DEVICE, USE_CHANNELS_LAST)


def get_model():
    from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

    model = PDETransformer(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        type=MODEL_TYPE,
        patch_size=4,
        periodic=False,
        carrier_token_active=True,
    ).to(DEVICE)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model


def create_optimizer(model):
    adamw_kwargs = {
        "lr": LR,
        "weight_decay": 1e-6,
        "foreach": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        with suppress(TypeError, RuntimeError):
            adamw_kwargs["fused"] = True
            return torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        adamw_kwargs.pop("fused", None)
    return torch.optim.AdamW(model.parameters(), **adamw_kwargs)


def move_batch_to_device(x, y, mask):
    if DEVICE != "cuda":
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y = y.to(DEVICE, non_blocking=PIN_MEMORY)
        mask = mask.to(DEVICE, non_blocking=PIN_MEMORY)
        if USE_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.contiguous(memory_format=torch.channels_last)
            mask = mask.contiguous(memory_format=torch.channels_last)
    return x, y, mask


def update_progress(progress_bar, total_loss_sum, mse_sum, spec_sum, sample_count):
    progress_bar.set_postfix(
        tot=f"{total_loss_sum / max(1, sample_count):.6f}",
        mse=f"{mse_sum / max(1, sample_count):.6f}",
        sp=f"{spec_sum / max(1, sample_count):.6f}",
    )


def build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val": best_metric,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    checkpoint = build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses)
    torch.save(checkpoint, os.path.join(OUT_DIR, "last.ckpt"))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        saved_epoch = int(checkpoint.get("epoch", 0))
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return {
            "resumed": True,
            "saved_epoch": saved_epoch,
            "best_val": float(checkpoint.get("best_val", math.inf)),
            "train_losses": list(checkpoint.get("train_losses", [])),
            "val_losses": list(checkpoint.get("val_losses", [])),
        }

    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        return {
            "resumed": False,
            "saved_epoch": None,
            "best_val": math.inf,
            "train_losses": [],
            "val_losses": [],
        }

    raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")


def run_epoch(model, loader, zero_norm, get_labels_fn, training, optimizer=None):
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        desc = "tr"
    else:
        model.eval()
        desc = "va"

    loss_sum = 0.0
    mse_sum = 0.0
    spec_sum = 0.0
    sample_count = 0

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        progress_bar = make_progress(loader, desc)
        for step, (x, y, mask) in enumerate(progress_bar):
            x, y, mask = move_batch_to_device(x, y, mask)
            labels = get_labels_fn(x.shape[0])

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                pred = model(x, class_labels=labels).sample
            pred = pred.float()
            pred = torch.lerp(zero_norm, pred, mask)

            raw_loss, mse_term, spec_term = hybrid_spatial_spectral_loss(pred, y)

            if training:
                (raw_loss / ACCUM_GRAD).backward()
                if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = x.shape[0]
            loss_sum += raw_loss.item() * batch_size
            mse_sum += mse_term.item() * batch_size
            spec_sum += spec_term.item() * batch_size
            sample_count += batch_size
            if step == 0 or (step + 1) % TQDM_UPDATE_EVERY == 0 or (step + 1) == len(loader):
                update_progress(progress_bar, loss_sum, mse_sum, spec_sum, sample_count)

    return (
        loss_sum / max(1, sample_count),
        mse_sum / max(1, sample_count),
        spec_sum / max(1, sample_count),
    )


def save_rollout_video(model_to_render, sim_info, mean, std, zero_norm, get_labels_fn, out_path, title_tag):
    video_states = load_packed_array(sim_info["states_path"])
    warmup_idx = warmup_start_index(sim_info["n_frames"])
    usable_len = max(1, sim_info["n_frames"] - warmup_idx)
    start_idx = warmup_idx + int(usable_len * 0.50)
    end_idx = warmup_idx + int(usable_len * 0.75)
    end_idx = min(sim_info["n_frames"], max(start_idx + 1, end_idx))
    rollout_len = end_idx - start_idx

    gt_frames = np.asarray(video_states[start_idx:end_idx], dtype=np.float32)

    mean_np = mean.numpy()[:, None, None]
    std_np = std.numpy()[:, None, None]
    gt_norm = (gt_frames - mean_np) / std_np

    mask_np = np.asarray(load_packed_array(sim_info["packed_mask_path"]), dtype=np.float32)
    mask_tensor = torch.from_numpy(mask_np).float().to(DEVICE).unsqueeze(0)

    pred_frames = [gt_norm[0]]
    model_to_render.eval()
    with torch.inference_mode():
        current = torch.tensor(gt_norm[0], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        labels = get_labels_fn(1)
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}/")

    print("\nDiscovering simulation folders …")
    sim_infos = discover_simulations(SIM_ROOT)
    print(f"Found {len(sim_infos)} simulations under {SIM_ROOT}")
    total_frames = sum(sim["n_frames"] for sim in sim_infos)
    print(f"Total frames across all simulations: {total_frames}")

    print("\nEnsuring packed simulation caches …")
    ensure_all_sim_caches(sim_infos, CACHE_WORKERS, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)

    train_sim_infos, val_sim_infos = split_simulations(sim_infos, VAL_FRAC)
    print(f"Train simulations: {len(train_sim_infos)}   Val simulations: {len(val_sim_infos)}")

    print("Computing normalization statistics …")
    mean, std = compute_global_stats(train_sim_infos, sim_infos)
    print(f"  per-channel mean: {mean.numpy().round(5)}")
    print(f"  per-channel std:  {std.numpy().round(5)}")

    train_ds = MultiSimKarmanDataset(train_sim_infos, mean, std)
    val_ds = MultiSimKarmanDataset(val_sim_infos, mean, std) if val_sim_infos else None
    train_loader = maybe_wrap_prefetch(build_loader(train_ds, shuffle=True))
    val_loader = maybe_wrap_prefetch(build_loader(val_ds, shuffle=False)) if val_ds is not None else None
    print(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds) if val_ds is not None else 0}")

    zero_norm = ((torch.zeros(3, device=DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)).view(1, 3, 1, 1)
    if USE_CHANNELS_LAST:
        zero_norm = zero_norm.contiguous(memory_format=torch.channels_last)

    print("\nBuilding PDE-Transformer (MC-S) …")
    model = get_model()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")
    if USE_CHANNELS_LAST:
        print("  Memory format: channels_last")
    if DEVICE == "cuda":
        print("  Device prefetch: enabled")

    optimizer = create_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    task_label = torch.tensor([1000], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    train_losses = []
    val_losses = []
    train_mse_terms = []
    train_spec_terms = []
    val_mse_terms = []
    val_spec_terms = []
    best_val = math.inf
    loss_log_path = None
    start_epoch = 1

    resume_path = RESUME_CHECKPOINT
    if resume_path and os.path.exists(resume_path):
        checkpoint_info = load_checkpoint(model, optimizer, scheduler, resume_path)
        resumed = checkpoint_info["resumed"]
        saved_epoch = checkpoint_info["saved_epoch"]
        best_val = checkpoint_info["best_val"]
        train_losses = checkpoint_info["train_losses"]
        val_losses = checkpoint_info["val_losses"]
        if saved_epoch is not None:
            start_epoch = saved_epoch + 1
        print(f"Loaded checkpoint: {resume_path}")
        if resumed:
            print(
                f"Successfully loaded checkpoint weights. "
                f"Checkpoint was saved at epoch {saved_epoch}; resuming at epoch {start_epoch}."
            )
        else:
            print("Successfully loaded model weights from checkpoint; optimizer/scheduler state not found.")

    if not SKIP_TRAIN:
        loss_log_path = os.path.join(OUT_DIR, f"hybrid_loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(loss_log_path, "w", encoding="utf-8") as log_file:
            log_file.write("timestamp epoch hybrid_loss train_mse train_spec val_loss val_mse val_spec best\n")

        for epoch in range(start_epoch, EPOCHS + 1):
            train_loss, train_mse_avg, train_spec_avg = run_epoch(
                model=model,
                loader=train_loader,
                zero_norm=zero_norm,
                get_labels_fn=get_labels,
                training=True,
                optimizer=optimizer,
            )

            if val_loader is None:
                val_loss = float("inf")
                val_mse_avg = float("nan")
                val_spec_avg = float("nan")
            else:
                val_loss, val_mse_avg, val_spec_avg = run_epoch(
                    model=model,
                    loader=val_loader,
                    zero_norm=zero_norm,
                    get_labels_fn=get_labels,
                    training=False,
                )

            scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_mse_terms.append(train_mse_avg)
            train_spec_terms.append(train_spec_avg)
            val_mse_terms.append(val_mse_avg)
            val_spec_terms.append(val_spec_avg)

            if val_loss < best_val:
                best_val = val_loss
                best_marker = " ← best"
            else:
                best_marker = ""

            save_checkpoint(model, optimizer, scheduler, epoch, best_val, train_losses, val_losses)

            if epoch % 2 == 0:
                epoch_video_path = os.path.join(OUT_DIR, f"pred_vs_gt_epoch_{epoch:03d}.mp4")
                video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
                save_rollout_video(model, video_sim, mean, std, zero_norm, get_labels, epoch_video_path, f"Epoch {epoch:03d}")

            print(
                f"Epoch {epoch:02d}/{EPOCHS}  comprehensive_loss={train_loss:.6f}  "
                f"train_mse={train_mse_avg:.6f}  train_spec={train_spec_avg:.6f}  "
                f"val_loss={val_loss:.6f}  val_mse={val_mse_avg:.6f}  val_spec={val_spec_avg:.6f}{best_marker}"
            )

            epoch_log_note = "best" if best_marker else ""
            if loss_log_path:
                log_line = (
                    f"{datetime.now().isoformat()} epoch={epoch} hybrid_loss={train_loss:.6f} "
                    f"train_mse={train_mse_avg:.6f} train_spec={train_spec_avg:.6f} "
                    f"val_loss={val_loss:.6f} val_mse={val_mse_avg:.6f} val_spec={val_spec_avg:.6f} {epoch_log_note}\n"
                )
                with open(loss_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(log_line)

        print("\nSaving loss curve …")
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="#111")
        ax.set_facecolor("#111")
        ax.plot(range(1, EPOCHS + 1), train_losses, color="#00c8ff", linewidth=2, label="Train Hybrid Loss")
        ax.plot(range(1, EPOCHS + 1), val_losses, color="#ff6b35", linewidth=2, label="Val Hybrid Loss")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Hybrid Loss", color="white")
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
    video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
    frame_path = save_rollout_video(model, video_sim, mean, std, zero_norm, get_labels, os.path.join(OUT_DIR, "pred_vs_gt.mp4"), "Final")
    if not SKIP_TRAIN:
        print(f"\n{'=' * 60}")
        print("Training complete.")
        print(f"  Best val loss : {best_val:.6f}")
        if loss_log_path:
            print(f"  Hybrid loss log : {loss_log_path}")
        print(f"  Loss curve   : {OUT_DIR}/loss_curve.png")
        print(f"  Prediction   : {frame_path}")
        print(f"  Checkpoint   : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 60}\n")
        header = (
            f"{'Epoch':>6}  {'Train Loss':>12}  {'Train MSE':>12}  {'Train Spec':>12}  "
            f"{'Val Loss':>12}  {'Val MSE':>12}  {'Val Spec':>12}"
        )
        print(header)
        print("-" * len(header))
        for epoch_idx, (hybrid, t_mse, t_spec, v_loss, v_mse, v_spec) in enumerate(
            zip(train_losses, train_mse_terms, train_spec_terms, val_losses, val_mse_terms, val_spec_terms), 1
        ):
            print(
                f"{epoch_idx:>6}  {hybrid:>12.6f}  {t_mse:>12.6f}  {t_spec:>12.6f}  {v_loss:>12.6f}  "
                f"{v_mse:>12.6f}  {v_spec:>12.6f}"
            )
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
