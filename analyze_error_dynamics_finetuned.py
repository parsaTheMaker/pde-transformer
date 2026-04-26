"""
analyze_error_dynamics_finetuned.py
===================================
Analyze absolute MSE, error velocity, and error acceleration of a fine-tuned
PDE-Transformer over a long autoregressive rollout.

This script performs no training. It instantiates the same PDE-Transformer +
LoRA architecture used by fine_tune_velocity_bptt_noCalib3.py, loads a base
checkpoint first when available, then overlays a fine-tuned checkpoint and
surveys long-rollout error dynamics.

Important:
  - The analysis rollout length is intentionally fixed by this script's
    default value (15) and does not inherit the shorter rollout used during
    fine-tuning.
"""

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from contextlib import suppress
from multiprocessing import cpu_count, freeze_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from peft import LoraConfig, get_peft_model

from sim_cache import discover_simulations, ensure_all_sim_caches, load_packed_array


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_ANALYSIS_ROLLOUT_LEN = 15
DEFAULT_BATCH_SIZE = 24
DEFAULT_NUM_ANALYSIS_BATCHES = 50
DEFAULT_SIM_ROOT = "./data/256_inc"
DEFAULT_OUT_DIR = os.path.join("runs", "analysis_finetuned_error_dynamics")
DEFAULT_BASE_CHECKPOINT = "./runs/karman_mse/last.ckpt"
DEFAULT_FINETUNED_CHECKPOINT = "./runs/karman_finetuned_velocity_LoRA_bptt_noCalib_curriculum3/last.ckpt"
DEFAULT_ANALYSIS_SPLIT = "val"
VAL_FRAC = 0.10
WARMUP_FRAC = 0.50

MODEL_TYPE = "PDE-S"
MODEL_SAMPLE_SIZE = 256
MODEL_IN_CHANNELS = 3
MODEL_OUT_CHANNELS = 3
MODEL_PATCH_SIZE = 4
MODEL_PERIODIC = False
MODEL_CARRIER_TOKEN_ACTIVE = True
TASK_LABEL_ID = 1000

LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["qkv", "to_qkv", "fc1", "fc2"]
LORA_DROPOUT = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_CHANNELS_LAST = False
NUM_WORKERS = max(0, cpu_count() - 5)
PIN_MEMORY = DEVICE == "cuda"
CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
CACHE_WORKERS = max(1, cpu_count() - 5)
PREFETCH_FACTOR = 2
NORM_STD_EPS = 1e-6


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------
if os.name != "nt":
    os.environ["LD_LIBRARY_PATH"] = (
        "/home/vatani/miniconda/envs/ag_env/lib/python3.11/site-packages/nvidia/cudnn/lib:" +
        os.environ.get("LD_LIBRARY_PATH", "")
    )

torch.backends.cudnn.enabled = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    with suppress(AttributeError):
        torch.backends.cuda.matmul.allow_tf32 = True
    with suppress(AttributeError):
        torch.backends.cudnn.allow_tf32 = True
    with suppress(AttributeError):
        torch.set_float32_matmul_precision("high")

torch.manual_seed(42)
np.random.seed(42)

try:
    import torch.nn.attention.flex_attention
    if not hasattr(torch.nn.attention.flex_attention, "AuxRequest"):
        class AuxRequest:
            pass
        torch.nn.attention.flex_attention.AuxRequest = AuxRequest
except (ImportError, AttributeError):
    pass


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def packed_slice_to_numpy(array):
    return np.array(array, dtype=np.float32, copy=True)


def warmup_start_index(num_frames, warmup_frac=WARMUP_FRAC):
    return int(num_frames * warmup_frac)


def split_simulations(sim_infos, val_frac):
    if len(sim_infos) <= 1:
        return sim_infos, []
    val_count = max(1, int(round(len(sim_infos) * val_frac)))
    val_count = min(val_count, len(sim_infos) - 1)
    return sim_infos[:-val_count], sim_infos[-val_count:]


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
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + NORM_STD_EPS
    return mean, std


class MultiSimKarmanDataset(Dataset):
    def __init__(self, sim_list, mean, std, max_rollout):
        self.sims = sim_list
        self.mean_np = mean[:, None, None].numpy().astype(np.float32, copy=True)
        self.inv_std_np = (1.0 / std[:, None, None].numpy()).astype(np.float32, copy=True)
        self.samples = []
        self._state_arrays = {}
        self._mask_tensors = {}
        self.max_rollout = max_rollout

        for sim_idx, sim in enumerate(self.sims):
            start_idx = warmup_start_index(sim["n_frames"])
            safe_end = sim["n_frames"] - 1 - self.max_rollout
            safe_end = max(safe_end, start_idx + 1)
            for frame_idx in range(start_idx, safe_end):
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
        np.subtract(x_np, self.mean_np, out=x_np)
        np.multiply(x_np, self.inv_std_np, out=x_np)

        end_idx = frame_idx + 1 + self.max_rollout
        y_seq_np = packed_slice_to_numpy(states[frame_idx + 1: end_idx])
        y_seq_np = np.asarray(y_seq_np, dtype=np.float32)
        np.subtract(y_seq_np, self.mean_np, out=y_seq_np)
        np.multiply(y_seq_np, self.inv_std_np, out=y_seq_np)

        return (
            torch.from_numpy(x_np).float(),
            torch.from_numpy(y_seq_np).float(),
            self._get_mask(sim_idx),
        )


def build_loader(dataset, batch_size, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
        if os.name != "nt":
            kwargs["multiprocessing_context"] = "fork"
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
        x, y_seq, mask = batch
        x = x.to(self.device, non_blocking=True)
        y_seq = y_seq.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)
        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            y_seq = y_seq.contiguous(memory_format=torch.channels_last)
            mask = mask.contiguous(memory_format=torch.channels_last)
        return x, y_seq, mask

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


def maybe_wrap_prefetch(loader):
    if loader is None or DEVICE != "cuda":
        return loader
    return DevicePrefetchLoader(loader, DEVICE, USE_CHANNELS_LAST)


def move_batch_to_device(x, y_seq, mask):
    if x.device.type != DEVICE:
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
    if y_seq.device.type != DEVICE:
        y_seq = y_seq.to(DEVICE, non_blocking=PIN_MEMORY)
    if mask.device.type != DEVICE:
        mask = mask.to(DEVICE, non_blocking=PIN_MEMORY)
    if USE_CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
        y_seq = y_seq.contiguous(memory_format=torch.channels_last)
        mask = mask.contiguous(memory_format=torch.channels_last)
    return x, y_seq, mask


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def get_model(base_checkpoint_path):
    from pdetransformer.core.mixed_channels.pde_transformer import PDETransformer

    base_model = PDETransformer(
        sample_size=MODEL_SAMPLE_SIZE,
        in_channels=MODEL_IN_CHANNELS,
        out_channels=MODEL_OUT_CHANNELS,
        type=MODEL_TYPE,
        patch_size=MODEL_PATCH_SIZE,
        periodic=MODEL_PERIODIC,
        carrier_token_active=MODEL_CARRIER_TOKEN_ACTIVE,
    )

    if base_checkpoint_path and os.path.exists(base_checkpoint_path):
        checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        base_model.load_state_dict(state_dict, strict=False)

    base_model = base_model.to(DEVICE)
    if USE_CHANNELS_LAST:
        base_model = base_model.to(memory_format=torch.channels_last)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model = peft_model.to(DEVICE)

    if USE_CHANNELS_LAST:
        peft_model = peft_model.to(memory_format=torch.channels_last)

    return peft_model


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)


# ---------------------------------------------------------------------------
# Data collection and plotting
# ---------------------------------------------------------------------------
def collect_error_dynamics(model, loader, zero_norm, get_labels_fn, rollout_len, num_batches):
    print(f"\nCollecting error dynamics over {num_batches} batches...")
    model.eval()
    all_mses = []

    total_batches = min(len(loader), num_batches)

    with torch.inference_mode():
        for batch_idx, (x_batch, y_seq, mask) in enumerate(
            tqdm(loader, desc="Analyzing", total=total_batches, leave=True, dynamic_ncols=True)
        ):
            if batch_idx >= num_batches:
                break

            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            max_rollout = min(rollout_len, y_seq.shape[1])
            state = x_batch.clone()
            batch_mses = []

            for t in range(max_rollout):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(state, class_labels=labels).sample
                pred = pred.float()

                mse_per_sample = F.mse_loss(pred, y_seq[:, t], reduction="none").mean(dim=(1, 2, 3))
                batch_mses.append(mse_per_sample.cpu().numpy())

                state = torch.lerp(zero_norm, pred, mask)

            all_mses.append(np.stack(batch_mses, axis=1))

    if not all_mses:
        return np.empty((0, rollout_len), dtype=np.float32)
    return np.concatenate(all_mses, axis=0)


def analyze_and_plot(all_mses, out_dir, summary_title):
    os.makedirs(out_dir, exist_ok=True)

    if all_mses.size == 0:
        raise RuntimeError("No analysis samples were collected.")

    mean_mse = np.mean(all_mses, axis=0)
    velocities = all_mses[:, 1:] - all_mses[:, :-1]
    mean_vel = np.mean(velocities, axis=0)
    std_vel = np.std(velocities, axis=0)

    accels = all_mses[:, 2:] - 2.0 * all_mses[:, 1:-1] + all_mses[:, :-2]
    mean_accel = np.mean(accels, axis=0) if accels.shape[1] > 0 else np.empty((0,), dtype=np.float32)
    std_accel = np.std(accels, axis=0) if accels.shape[1] > 0 else np.empty((0,), dtype=np.float32)

    max_rollout = all_mses.shape[1]
    output_lines = [
        "",
        summary_title,
        f"{'Step':>6} | {'Mean MSE':>12} | {'Mean Vel':>12} | {'Std Vel':>12} | {'Mean Accel':>12} | {'Std Accel':>12}",
        "-" * 79,
    ]

    for t in range(max_rollout):
        mse_str = f"{mean_mse[t]:>12.6e}"

        if t < 1:
            vel_str, svel_str = f"{'N/A':>12}", f"{'N/A':>12}"
        else:
            vel_str = f"{mean_vel[t - 1]:>12.6e}"
            svel_str = f"{std_vel[t - 1]:>12.6e}"

        if t < 2:
            acc_str, sacc_str = f"{'N/A':>12}", f"{'N/A':>12}"
        else:
            acc_str = f"{mean_accel[t - 2]:>12.6e}"
            sacc_str = f"{std_accel[t - 2]:>12.6e}"

        output_lines.append(f"{t:>6} | {mse_str} | {vel_str} | {svel_str} | {acc_str} | {sacc_str}")

    summary_text = "\n".join(output_lines)
    print(summary_text)

    summary_path = os.path.join(out_dir, "error_dynamics_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    raw_path = os.path.join(out_dir, "error_dynamics_raw_mse.npy")
    np.save(raw_path, all_mses.astype(np.float32, copy=False))

    steps = np.arange(max_rollout)
    vel_steps = np.arange(1, max_rollout)
    accel_steps = np.arange(2, max_rollout)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#111")
    fig.suptitle(summary_title, color="white", fontsize=16)

    ax = axes[0, 0]
    ax.set_facecolor("#111")
    ax.plot(steps, mean_mse, marker="o", color="#00c8ff", linewidth=2)
    ax.set_title("Mean Absolute MSE ($E_t$)", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("MSE", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    ax = axes[0, 1]
    ax.set_facecolor("#111")
    ax.plot(vel_steps, mean_vel, marker="o", color="#00ff88", linewidth=2)
    ax.set_title("Mean Velocity ($v_t$)", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("Mean Velocity", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    ax = axes[1, 0]
    ax.set_facecolor("#111")
    if accel_steps.size > 0:
        ax.plot(accel_steps, mean_accel, marker="o", color="#ffaa00", linewidth=2)
    ax.set_title("Mean Acceleration ($a_t$)", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("Mean Acceleration", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    ax = axes[1, 1]
    ax.set_facecolor("#111")
    ax.plot(vel_steps, std_vel, marker="o", color="#00ff88", linestyle="--", label="Std Velocity")
    if accel_steps.size > 0:
        ax.plot(accel_steps, std_accel, marker="o", color="#ffaa00", linestyle="--", label="Std Acceleration")
    ax.set_title("Standard Deviations", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("Std Dev", color="white")
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "error_dynamics_analysis.png")
    plt.savefig(plot_path, dpi=150, facecolor="#111")
    plt.close()

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw MSEs to: {raw_path}")
    print(f"Saved plot to: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze long-rollout error dynamics of a fine-tuned PDE-Transformer.")
    parser.add_argument("--sim-root", default=DEFAULT_SIM_ROOT, help="Root directory containing simulation folders.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory where analysis outputs are written.")
    parser.add_argument(
        "--base-checkpoint",
        default=DEFAULT_BASE_CHECKPOINT,
        help="Optional base checkpoint loaded before the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        default=DEFAULT_FINETUNED_CHECKPOINT,
        help="Fine-tuned checkpoint to analyze.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default=DEFAULT_ANALYSIS_SPLIT,
        help="Which simulation split to analyze.",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=DEFAULT_ANALYSIS_ROLLOUT_LEN,
        help="Autoregressive rollout length used for analysis. Defaults to 15.",
    )
    parser.add_argument(
        "--analysis-batches",
        type=int,
        default=DEFAULT_NUM_ANALYSIS_BATCHES,
        help="Number of batches to survey.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Analysis batch size. Reduce this if you hit GPU OOM.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}/")
    print(f"Analysis rollout length: {args.rollout_len}")
    print(f"Analysis split: {args.split}")

    print("\nDiscovering simulation folders ...")
    sim_infos = discover_simulations(args.sim_root)
    print(f"Found {len(sim_infos)} simulations under {args.sim_root}")

    print("\nEnsuring packed simulation caches ...")
    ensure_all_sim_caches(sim_infos, CACHE_WORKERS, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)

    train_sim_infos, val_sim_infos = split_simulations(sim_infos, VAL_FRAC)
    print(f"Train simulations: {len(train_sim_infos)}   Val simulations: {len(val_sim_infos)}")

    print("Computing normalization statistics from train simulations ...")
    mean, std = compute_global_stats(train_sim_infos, sim_infos)
    print(f"  per-channel mean: {mean.numpy().round(5)}")
    print(f"  per-channel std:  {std.numpy().round(5)}")

    if args.split == "val":
        analysis_sim_infos = val_sim_infos if val_sim_infos else train_sim_infos
        if not val_sim_infos:
            print("Validation split is empty; falling back to train split for analysis.")
    else:
        analysis_sim_infos = train_sim_infos

    analysis_ds = MultiSimKarmanDataset(
        analysis_sim_infos,
        mean,
        std,
        max_rollout=args.rollout_len,
    )
    analysis_loader = maybe_wrap_prefetch(
        build_loader(analysis_ds, batch_size=args.batch_size, shuffle=(args.split == "train"))
    )
    print(f"Analysis samples: {len(analysis_ds)}")

    zero_norm = ((torch.zeros(MODEL_IN_CHANNELS, device=DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)).view(1, MODEL_IN_CHANNELS, 1, 1)
    if USE_CHANNELS_LAST:
        zero_norm = zero_norm.contiguous(memory_format=torch.channels_last)

    print("\nBuilding PDE-Transformer + LoRA analysis model ...")
    model = get_model(args.base_checkpoint)

    if args.finetuned_checkpoint and os.path.exists(args.finetuned_checkpoint):
        load_model_checkpoint(model, args.finetuned_checkpoint)
        print(f"Loaded fine-tuned checkpoint: {args.finetuned_checkpoint}")
    else:
        print("WARNING: Fine-tuned checkpoint not found. Analyzing base-initialized model instead.")

    task_label = torch.tensor([TASK_LABEL_ID], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    all_mses = collect_error_dynamics(
        model=model,
        loader=analysis_loader,
        zero_norm=zero_norm,
        get_labels_fn=get_labels,
        rollout_len=args.rollout_len,
        num_batches=args.analysis_batches,
    )

    analyze_and_plot(
        all_mses,
        args.out_dir,
        summary_title=f"Error Dynamics over Long Rollout ({args.split} split, rollout={args.rollout_len})",
    )


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
