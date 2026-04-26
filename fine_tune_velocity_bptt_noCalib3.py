"""
fine_tune_velocity_bptt_noCalib3.py
=====================
Fine-tune the frozen PDE-Transformer base model using LoRA (via PEFT) on
autoregressive multi-step rollout with an epoch-local rollout curriculum
for long-rollout stability.

Key design decisions:
  - MSE loss is computed on raw model output (no mask in loss).
  - The mask is applied ONLY when advancing the rollout state.
    - Every epoch starts at curriculum frontier N=1 (second rollout step).
        - Promotion is stage-based and uses a persisted required-improvement threshold.
            The threshold is adapted between epochs from rollout reachability and checkpointed.
    - Full BPTT is applied through all steps 0..N with equal-weight mean loss.
    - Curriculum stage state resets on every promotion and at each epoch.
    - LoRA is injected into qkv, to_qkv, fc1, fc2 via PEFT.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Rollout curriculum configuration
# ---------------------------------------------------------------------------
MAX_ROLLOUT_LEN = 8

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

# ---------------------------------------------------------------------------
# Distributed (DDP) globals
# ---------------------------------------------------------------------------
RANK = 0
LOCAL_RANK = 0
WORLD_SIZE = 1
DDP_ENABLED = False


def is_main_process():
    return RANK == 0


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    global RANK, LOCAL_RANK, WORLD_SIZE, DDP_ENABLED

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        RANK = int(os.environ["RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        DDP_ENABLED = WORLD_SIZE > 1
    else:
        RANK = 0
        LOCAL_RANK = 0
        WORLD_SIZE = 1
        DDP_ENABLED = False

    if DDP_ENABLED:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend=backend, init_method="env://")


def cleanup_distributed():
    if DDP_ENABLED and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ddp_barrier():
    if DDP_ENABLED and dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available() and DEVICE == "cuda":
            dist.barrier(device_ids=[LOCAL_RANK])
        else:
            dist.barrier()


def prepare_sim_cache_metadata(sim_infos):
    for sim_info in sim_infos:
        if "states_path" not in sim_info or "packed_mask_path" not in sim_info:
            from sim_cache import prepare_sim_cache_info
            prepare_sim_cache_info(sim_info, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def gather_val_mses(local_arr):
    if not DDP_ENABLED:
        return local_arr

    local_np = np.asarray(local_arr, dtype=np.float32)
    if local_np.ndim != 2:
        if local_np.size == 0:
            local_np = np.empty((0, 0), dtype=np.float32)
        else:
            local_np = local_np.reshape(local_np.shape[0], -1)

    local_rows, local_cols = local_np.shape
    local_shape = torch.tensor([local_rows, local_cols], dtype=torch.int64, device=DEVICE)
    gathered_shapes = [torch.zeros(2, dtype=torch.int64, device=DEVICE) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_shapes, local_shape)

    rows_per_rank = [int(t[0].item()) for t in gathered_shapes]
    cols_per_rank = [int(t[1].item()) for t in gathered_shapes]
    max_rows = max(rows_per_rank) if rows_per_rank else 0
    max_cols = max(cols_per_rank) if cols_per_rank else 0

    padded_local = torch.zeros((max_rows, max_cols), dtype=torch.float32, device=DEVICE)
    if local_rows > 0 and local_cols > 0:
        local_tensor = torch.as_tensor(local_np, dtype=torch.float32, device=DEVICE)
        padded_local[:local_rows, :local_cols] = local_tensor

    gathered = [torch.empty_like(padded_local) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, padded_local)

    if not is_main_process():
        return None

    arrays = []
    for rank_tensor, rows, cols in zip(gathered, rows_per_rank, cols_per_rank):
        if rows > 0 and cols > 0:
            arrays.append(rank_tensor[:rows, :cols].cpu().numpy())

    if not arrays:
        return None

    target_cols = max(a.shape[1] for a in arrays)
    if any(a.shape[1] != target_cols for a in arrays):
        aligned = []
        for arr in arrays:
            if arr.shape[1] == target_cols:
                aligned.append(arr)
            else:
                padded = np.full((arr.shape[0], target_cols), np.nan, dtype=np.float32)
                padded[:, :arr.shape[1]] = arr
                aligned.append(padded)
        arrays = aligned

    return np.concatenate(arrays, axis=0)


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

SIM_ROOT = "./data/256_inc"
OUT_DIR = os.path.join("runs", "karman_finetuned_velocity_LoRA_bptt_noCalib_curriculum3")
EPOCHS = 40
BATCH_SIZE = 8
ACCUM_GRAD = 1
LR = 5e-5
VAL_FRAC = 0.1
WARMUP_FRAC = 0.5
from sim_cache import discover_simulations, ensure_all_sim_caches, load_packed_array

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS_VID = 10
VID_FRAMES = 50
DPI_VID = 110
SKIP_TRAIN = False
RESUME_CHECKPOINT = "./runs/karman_mse/last.ckpt"
MODEL_TYPE = "PDE-S"
MODEL_SAMPLE_SIZE = 256
MODEL_IN_CHANNELS = 3
MODEL_OUT_CHANNELS = 3
MODEL_PATCH_SIZE = 4
MODEL_PERIODIC = False
MODEL_CARRIER_TOKEN_ACTIVE = True

LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["qkv", "to_qkv", "fc1", "fc2"]
LORA_DROPOUT = 0.05

USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_CHANNELS_LAST = False
NUM_WORKERS = max(0, cpu_count() - 5)
# For DDP, default to zero worker processes per rank to minimize
# dataloader process and IPC overhead. Override via env as needed.
try:
    DDP_LOADER_WORKERS = max(0, int(os.environ.get("DDP_LOADER_WORKERS", "0")))
except ValueError:
    DDP_LOADER_WORKERS = 0
PIN_MEMORY = DEVICE == "cuda"
CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
CACHE_WORKERS = max(1, cpu_count() - 5)
PREFETCH_FACTOR = 2
TQDM_UPDATE_EVERY = 20
DDP_BUCKET_CAP_MB = 100
DDP_USE_STATIC_GRAPH = False
N_PROGRESS_EMA_ALPHA = 0.15

OPTIM_WEIGHT_DECAY = 1e-6
GRAD_CLIP_NORM = 1.0
SCHED_STEP_SIZE = 10
SCHED_GAMMA = 0.25
TASK_LABEL_ID = 1000

# Curriculum hyperparameter (initial required frontier-velocity improvement).
# The next-epoch value is adjusted from rollout reachability and persisted.
VELOCITY_IMPROVEMENT_FRAC = 0.30
# Hard minimum improvement requirement (1%).
MIN_IMPROVEMENT_FRAC = 0.01
# Method rule (fixed): require at least this many baseline batches before
# promotion checks.
MIN_STAGE_BATCHES = 30
# Method rule (fixed): promotion compares the average over the latest
# RECENT_PROMO_WINDOW frontier-reaching batches against the prior-stage
# baseline average.
RECENT_PROMO_WINDOW = 10
# Memory optimization: recompute training activations during backward to reduce VRAM.
USE_ACTIVATION_CHECKPOINTING = True

NORM_STD_EPS = 1e-6
WARMUP_DUMMY_BATCH = 1
VIDEO_START_FRAC = 0.50
VIDEO_END_FRAC = 0.75
VIDEO_CODEC = "libx264"
VIDEO_BITRATE = 1800


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def packed_slice_to_numpy(array):
    return np.array(array, dtype=np.float32, copy=True)


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
        if "states_path" not in sim:
            from sim_cache import prepare_sim_cache_info
            prepare_sim_cache_info(sim, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)
        states = load_packed_array(sim["states_path"])
        start_idx = warmup_start_index(sim["n_frames"])
        usable_frames = max(1, sim["n_frames"] - start_idx)
        sample_count = max(1, int(round(target_samples * (usable_frames / total_frames))))
        idxs = np.linspace(start_idx, sim["n_frames"] - 1, sample_count, dtype=int)
        samples.append(np.asarray(states[idxs], dtype=np.float32))

    if not samples:
        sim = source_sims[0]
        if "states_path" not in sim:
            from sim_cache import prepare_sim_cache_info
            prepare_sim_cache_info(sim, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)
        states = load_packed_array(sim["states_path"])
        samples.append(np.asarray(states[[0]], dtype=np.float32))

    stacked = np.concatenate(samples, axis=0)
    mean = torch.tensor(stacked.mean(axis=(0, 2, 3)), dtype=torch.float32)
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + NORM_STD_EPS
    return mean, std


class MultiSimKarmanDataset(Dataset):
    """Dataset that returns (x, y_seq, mask) where y_seq has shape [max_rollout, C, H, W]."""

    def __init__(self, sim_list, mean, std, max_rollout=MAX_ROLLOUT_LEN):
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
        x = torch.from_numpy(x_np)

        end_idx = frame_idx + 1 + self.max_rollout
        y_seq_np = packed_slice_to_numpy(states[frame_idx + 1: end_idx])
        y_seq_np = np.asarray(y_seq_np, dtype=np.float32)
        np.subtract(y_seq_np, self.mean_np, out=y_seq_np)
        np.multiply(y_seq_np, self.inv_std_np, out=y_seq_np)
        y_seq = torch.from_numpy(y_seq_np)

        return x.float(), y_seq.float(), self._get_mask(sim_idx)


def build_loader(dataset, shuffle, sampler=None):
    if DDP_ENABLED:
        # Keep loader overhead minimal in DDP by default; this can be raised
        # with DDP_LOADER_WORKERS for throughput tuning.
        per_rank_workers = min(NUM_WORKERS, DDP_LOADER_WORKERS)
    else:
        per_rank_workers = NUM_WORKERS

    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": per_rank_workers,
        "pin_memory": PIN_MEMORY,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if per_rank_workers > 0:
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


def make_progress(iterable, desc):
    return tqdm(
        iterable,
        desc=desc,
        leave=False,
        disable=not is_main_process(),
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
    )


def maybe_wrap_prefetch(loader):
    if loader is None or DEVICE != "cuda":
        return loader
    return DevicePrefetchLoader(loader, DEVICE, USE_CHANNELS_LAST)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def get_model():
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
    
    peft_resume = os.path.join(OUT_DIR, "last.ckpt")
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT) and not os.path.exists(peft_resume):
        try:
            checkpoint = torch.load(RESUME_CHECKPOINT, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            base_model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

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


def create_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    adamw_kwargs = {
        "lr": LR,
        "weight_decay": OPTIM_WEIGHT_DECAY,
        "foreach": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        with suppress(TypeError, RuntimeError):
            adamw_kwargs["fused"] = True
            return torch.optim.AdamW(params, **adamw_kwargs)
        adamw_kwargs.pop("fused", None)
    return torch.optim.AdamW(params, **adamw_kwargs)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
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


def update_progress(progress_bar, loss_dict, sample_count, n_count=0, frontier_vel_running=None, frontier_n=None):
    mse_denom = max(1, sample_count)
    mse_avg = loss_dict["mse"] / mse_denom
    mse_avg = mse_avg.detach().item() if isinstance(mse_avg, torch.Tensor) else float(mse_avg)

    if n_count > 0:
        n_avg = float(loss_dict["N"]) / float(n_count)
        n_str = f"{n_avg:.2e}"
    else:
        n_str = "N/A"

    vel_str = f"{float(frontier_vel_running):.6e}" if frontier_vel_running is not None else "N/A"
    frontier_n_str = str(int(frontier_n)) if frontier_n is not None else "N/A"

    progress_bar.set_postfix(
        mse=f"{mse_avg:.6e}",
        N=n_str,
        frontierN=frontier_n_str,
        frontier_vel=vel_str,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def build_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    train_losses,
    val_losses,
    promotion_improvement_frac_next,
):
    base_model = unwrap_model(model)
    return {
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val": best_metric,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "promotion_improvement_frac_next": float(promotion_improvement_frac_next),
    }


def normalize_loss_history(losses):
    if not isinstance(losses, dict):
        return {"mse": []}
    return {"mse": list(losses.get("mse", []))}


def per_sample_mse(pred, target):
    return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))


def model_predict_sample(model, state, labels, training):
    """Forward pass helper with optional activation checkpointing for training memory savings."""
    if training and USE_ACTIVATION_CHECKPOINTING:
        def _forward(inp, lbl):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                out = model(inp, class_labels=lbl).sample
            return out.float()

        with suppress(TypeError):
            return activation_checkpoint(
                _forward,
                state,
                labels,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return activation_checkpoint(
            _forward,
            state,
            labels,
            use_reentrant=False,
        )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
        pred_raw = model(state, class_labels=labels).sample
    return pred_raw.float()


def positive_velocity_sum_count(curr_err_per_sample, prev_err_per_sample):
    vel = curr_err_per_sample - prev_err_per_sample
    pos_vel = vel[vel > 0]
    return float(pos_vel.sum().item()), int(pos_vel.numel())


def global_positive_velocity(curr_err_per_sample, prev_err_per_sample):
    local_sum, local_count = positive_velocity_sum_count(curr_err_per_sample, prev_err_per_sample)
    stats = torch.tensor([local_sum, float(local_count)], dtype=torch.float64, device=DEVICE)
    if DDP_ENABLED:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    global_sum = float(stats[0].item())
    global_count = int(stats[1].item())
    global_vel = global_sum / float(global_count) if global_count > 0 else 0.0
    return global_vel, global_sum, global_count


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    train_losses,
    val_losses,
    promotion_improvement_frac_next,
):
    checkpoint = build_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        best_metric,
        train_losses,
        val_losses,
        promotion_improvement_frac_next,
    )
    torch.save(checkpoint, os.path.join(OUT_DIR, "last.ckpt"))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    unwrap_model(model).load_state_dict(state_dict, strict=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception:
                pass
        return {
            "resumed": True,
            "saved_epoch": int(checkpoint.get("epoch", 0)),
            "best_val": float(checkpoint.get("best_val", math.inf)),
            "train_losses": checkpoint.get("train_losses", {"mse": []}),
            "val_losses":   checkpoint.get("val_losses",   {"mse": []}),
            "promotion_improvement_frac_next": float(
                checkpoint.get("promotion_improvement_frac_next", VELOCITY_IMPROVEMENT_FRAC)
            ),
        }

    return {
        "resumed": False,
        "saved_epoch": None,
        "best_val": math.inf,
        "train_losses": {"mse": []},
        "val_losses": {"mse": []},
        "promotion_improvement_frac_next": float(VELOCITY_IMPROVEMENT_FRAC),
    }


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------
def run_epoch(
    model,
    loader,
    zero_norm,
    get_labels_fn,
    training,
    optimizer=None,
    epoch=None,
    total_epochs=None,
    promotion_improvement_frac=None,
):
    """Run one full epoch.

    MSE loss contract:
        loss = F.mse_loss(pred_raw, y_t)

    Masking (obstacle enforcement):
        state = torch.lerp(zero_norm, pred_raw, mask)

    Epoch-local rollout curriculum (training only):
        - Every epoch starts at frontier N=1 (second rollout step).
        - Each batch trains with full BPTT on steps 0..N using equal-weight mean loss.
                - Promotion is stage-based and uses the provided
                    `promotion_improvement_frac` threshold for this epoch.
                - Promotion condition checks mean(last RECENT_PROMO_WINDOW frontier
                    velocities) against mean(prior-stage frontier velocities), requiring
                    at least MIN_STAGE_BATCHES prior-stage batches before evaluation.
    """
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        desc = "tr"
    else:
        model.eval()
        desc = "va"

    loss_accum = {
        "mse":   torch.zeros((), device=DEVICE),
        "N":     0.0,
    }
    sample_count = 0
    n_count = 0
    frontier_vel_sum = 0.0
    frontier_vel_count = 0
    batch_progress = [] if (training and is_main_process()) else None
    batch_n_values = [] if (training and is_main_process()) else None
    all_val_mses = [] if not training else None
    max_rollout_seen = 0
    max_frontier_N = -1

    # Curriculum resets at the start of every epoch.
    current_target_N = 1
    stage_batch_count = 0
    stage_frontier_vels = []
    stage_vel_sum = 0.0
    active_stage_baseline_vel = None
    last_valid_stage_baseline_vel = None
    promotion_count = 0
    running_stage_vel = None

    if training:
        if promotion_improvement_frac is None:
            improvement_frac = float(VELOCITY_IMPROVEMENT_FRAC)
        else:
            improvement_frac = float(promotion_improvement_frac)
    else:
        improvement_frac = float("nan")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    n_batches = len(loader)

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        progress_bar = make_progress(loader, desc)
        for step, (x_batch, y_seq, mask) in enumerate(progress_bar):
            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])
            batch_size = x_batch.shape[0]

            max_rollout = min(MAX_ROLLOUT_LEN, y_seq.shape[1])
            max_rollout_seen = max(max_rollout_seen, max_rollout)

            if mask.ndim == 3:   
                mask = mask.unsqueeze(1)   

            if training:
                effective_target_N = min(current_target_N, max_rollout - 1)
                state = x_batch
                prev_err_per_sample = None
                loss_sum = torch.zeros((), device=DEVICE)
                batch_frontier_vel = None

                for t in range(effective_target_N + 1):
                    pred_raw = model_predict_sample(model, state, labels, training=True)

                    y_t = y_seq[:, t]
                    loss_sum = loss_sum + F.mse_loss(pred_raw, y_t)

                    # Curriculum statistics are detached from autograd to avoid graph retention.
                    mse_per_sample = per_sample_mse(pred_raw, y_t).detach()

                    if t == effective_target_N and t >= 1 and prev_err_per_sample is not None:
                        frontier_vel, _, _ = global_positive_velocity(mse_per_sample, prev_err_per_sample)
                        batch_frontier_vel = float(frontier_vel)

                    prev_err_per_sample = mse_per_sample
                    if t < effective_target_N:
                        state = torch.lerp(zero_norm, pred_raw, mask)

                loss_to_backprop = loss_sum / float(effective_target_N + 1)

                if batch_n_values is not None:
                    batch_progress.append(float(step + 1) / float(max(1, n_batches)))
                    batch_n_values.append(float(effective_target_N))

                (loss_to_backprop / ACCUM_GRAD).backward()

                if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == n_batches:
                    torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_accum["mse"] += loss_to_backprop.detach() * batch_size
                loss_accum["N"] += effective_target_N * batch_size
                n_count += batch_size
                max_frontier_N = max(max_frontier_N, effective_target_N)
                sample_count += batch_size

                if batch_frontier_vel is not None:
                    frontier_vel_sum += float(batch_frontier_vel)
                    frontier_vel_count += 1

                if (
                    batch_frontier_vel is not None
                    and effective_target_N == current_target_N
                ):
                    stage_frontier_vels.append(float(batch_frontier_vel))
                    stage_batch_count += 1
                    stage_vel_sum += float(batch_frontier_vel)
                    running_stage_vel = stage_vel_sum / float(stage_batch_count)

                    active_stage_baseline_vel = None
                    can_check_promotion = stage_batch_count >= (MIN_STAGE_BATCHES + RECENT_PROMO_WINDOW)
                    if can_check_promotion:
                        baseline_values = stage_frontier_vels[:-RECENT_PROMO_WINDOW]
                        recent_values = stage_frontier_vels[-RECENT_PROMO_WINDOW:]
                        if len(baseline_values) >= MIN_STAGE_BATCHES:
                            active_stage_baseline_vel = float(np.mean(baseline_values, dtype=np.float64))
                            last_valid_stage_baseline_vel = active_stage_baseline_vel
                            recent_mean_vel = float(np.mean(recent_values, dtype=np.float64))
                        else:
                            recent_mean_vel = None
                    else:
                        recent_mean_vel = None

                    if (
                        current_target_N < (MAX_ROLLOUT_LEN - 1)
                        and active_stage_baseline_vel is not None
                        and recent_mean_vel is not None
                        and recent_mean_vel <= active_stage_baseline_vel * (1.0 - improvement_frac)
                    ):
                        current_target_N += 1
                        promotion_count += 1
                        stage_batch_count = 0
                        stage_frontier_vels = []
                        stage_vel_sum = 0.0
                        active_stage_baseline_vel = None
                        running_stage_vel = None

            else:
                state = x_batch.clone()
                batch_mses = []
                avg_mse_accum = 0.0
                for t in range(max_rollout):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                        pred_raw = model(state, class_labels=labels).sample
                    pred_raw = pred_raw.float()

                    y_t = y_seq[:, t]
                    mse_loss = F.mse_loss(pred_raw, y_t)
                    avg_mse_accum += mse_loss
                    
                    mse_per_sample = per_sample_mse(pred_raw, y_t)
                    batch_mses.append(mse_per_sample.cpu().numpy())
                    state = torch.lerp(zero_norm, pred_raw, mask)
                
                loss_accum["mse"] += (avg_mse_accum / max_rollout).detach() * batch_size
                batch_mses_stacked = np.stack(batch_mses, axis=1)
                all_val_mses.append(batch_mses_stacked)
                sample_count += batch_size

            if step == 0 or (step + 1) % TQDM_UPDATE_EVERY == 0 or (step + 1) == n_batches:
                update_progress(
                    progress_bar,
                    loss_accum,
                    sample_count,
                    n_count=n_count,
                    frontier_vel_running=running_stage_vel if training else None,
                    frontier_n=current_target_N if training else None,
                )

    loss_tensor = torch.tensor(
        [
            float(loss_accum["mse"].detach().item()),
            float(loss_accum["N"]),
            float(n_count),
            float(sample_count),
            float(frontier_vel_sum),
            float(frontier_vel_count),
        ],
        device=DEVICE,
        dtype=torch.float64,
    )
    if DDP_ENABLED:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    max_target_tensor = torch.tensor([float(max_frontier_N)], device=DEVICE, dtype=torch.float64)
    if DDP_ENABLED:
        dist.all_reduce(max_target_tensor, op=dist.ReduceOp.MAX)

    if training:
        reported_baseline_vel = (
            active_stage_baseline_vel
            if active_stage_baseline_vel is not None
            else last_valid_stage_baseline_vel
        )
        scalar_state = torch.tensor(
            [
                float(current_target_N),
                float(stage_batch_count),
                float(stage_vel_sum),
                float(active_stage_baseline_vel) if active_stage_baseline_vel is not None else -1.0,
                float(reported_baseline_vel) if reported_baseline_vel is not None else -1.0,
                float(promotion_count),
                float(improvement_frac) if training else -1.0,
            ],
            device=DEVICE,
            dtype=torch.float64,
        )
        if DDP_ENABLED:
            dist.all_reduce(scalar_state, op=dist.ReduceOp.MAX)
        final_target_n = int(scalar_state[0].item())
        final_stage_batch_count = int(scalar_state[1].item())
        final_stage_vel_sum = float(scalar_state[2].item())
        final_active_stage_baseline = float(scalar_state[3].item())
        final_active_stage_baseline = (
            final_active_stage_baseline if final_active_stage_baseline >= 0.0 else None
        )
        final_reported_baseline = float(scalar_state[4].item())
        final_reported_baseline = final_reported_baseline if final_reported_baseline >= 0.0 else None
        total_promotions = int(scalar_state[5].item())
        final_improvement_frac = float(scalar_state[6].item())
    else:
        final_target_n = -1
        final_stage_batch_count = 0
        final_stage_vel_sum = 0.0
        final_active_stage_baseline = None
        final_reported_baseline = None
        total_promotions = 0
        final_improvement_frac = float("nan")

    total_mse, total_N, total_n_count, total_samples, total_frontier_vel_sum, total_frontier_vel_count = loss_tensor.tolist()
    denom = max(1.0, total_samples)
    max_N_reached = int(max_target_tensor.item())
    avg_target_n = (total_N / max(1.0, total_n_count)) if training else 0.0
    epoch_frontier_vel = (
        (total_frontier_vel_sum / total_frontier_vel_count)
        if (training and total_frontier_vel_count > 0)
        else float("nan")
    )

    if not training:
        local_val = (
            np.concatenate(all_val_mses, axis=0)
            if all_val_mses
            else np.empty((0, max_rollout_seen), dtype=np.float32)
        )
        gathered_val = gather_val_mses(local_val)
    else:
        gathered_val = None

    final_stage_running_vel = (
        final_stage_vel_sum / float(final_stage_batch_count)
        if final_stage_batch_count > 0
        else float("nan")
    )

    return {
        "mse": total_mse / denom,
        "avg_target_N": avg_target_n,
        "train_frontier_vel": epoch_frontier_vel,
        "frontier_active_stage_baseline_vel": final_active_stage_baseline,
        "frontier_baseline_vel": final_reported_baseline,
        "stage_batch_count": final_stage_batch_count,
        "promotion_count": total_promotions,
        "promotion_improvement_frac": final_improvement_frac,
        "train_frontier_N": final_target_n,
        "max_frontier_N": max_N_reached if training else max_rollout_seen - 1,
        "running_stage_frontier_vel": final_stage_running_vel,
        "frontier_vel_count": int(total_frontier_vel_count) if training else 0,
        "all_val_mses": gathered_val if (not training and is_main_process()) else None,
        "batch_progress": batch_progress if (training and is_main_process()) else None,
        "batch_n_values": batch_n_values if (training and is_main_process()) else None,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def analyze_and_plot(all_mses, out_dir, epoch, log_path=None):
    os.makedirs(out_dir, exist_ok=True)
    
    mean_mse = np.mean(all_mses, axis=0)
    MAX_ROLLOUT = all_mses.shape[1]
    
    velocities = all_mses[:, 1:] - all_mses[:, :-1]
    mean_vel = np.mean(velocities, axis=0)
    std_vel = np.std(velocities, axis=0)

    output_lines = [
        "",
        f"--- Epoch {epoch} Validation Error Dynamics ---",
        f"{'Step':>6} | {'Mean MSE':>12} | {'Mean Vel':>12} | {'Std Vel':>12}",
        "-" * 51
    ]
    
    for t in range(MAX_ROLLOUT):
        mse_str = f"{mean_mse[t]:>12.6e}"
        
        if t < 1:
            vel_str, svel_str = f"{'N/A':>12}", f"{'N/A':>12}"
        else:
            vel_str = f"{mean_vel[t-1]:>12.6e}"
            svel_str = f"{std_vel[t-1]:>12.6e}"

        output_lines.append(f"{t:>6} | {mse_str} | {vel_str} | {svel_str}")

    table_str = "\n".join(output_lines)
    print(table_str)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(table_str + "\n\n")

    steps = np.arange(MAX_ROLLOUT)
    vel_steps = np.arange(1, MAX_ROLLOUT)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111")
    fig.suptitle(f"Error Dynamics over Long Rollout - Epoch {epoch}", color="white", fontsize=16)

    ax = axes[0]
    ax.set_facecolor("#111")
    ax.plot(steps, mean_mse, marker='o', color="#00c8ff", linewidth=2)
    ax.set_title("Mean Absolute MSE ($E_t$)", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("MSE", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#555")

    ax = axes[1]
    ax.set_facecolor("#111")
    ax.plot(vel_steps, mean_vel, marker='o', color="#00ff88", linewidth=2)
    ax.set_title("Mean Velocity ($v_t$)", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("Mean Velocity", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#555")

    ax = axes[2]
    ax.set_facecolor("#111")
    ax.plot(vel_steps, std_vel, marker='o', color="#ffaa00", linewidth=2)
    ax.set_title("Std Velocity", color="white")
    ax.set_xlabel("Rollout Step (N)", color="white")
    ax.set_ylabel("Std Dev", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#555")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"error_dynamics_epoch_{epoch:03d}.png")
    plt.savefig(plot_path, dpi=150, facecolor="#111")
    plt.close()
    
    print(f"\nPlots saved successfully to: {plot_path}")


def plot_epoch_curriculum_n_progress_ema(
    batch_progress,
    batch_n_values,
    out_dir,
    epoch,
    ema_alpha=N_PROGRESS_EMA_ALPHA,
    log_path=None,
):
    os.makedirs(out_dir, exist_ok=True)

    x = np.asarray(batch_progress, dtype=np.float32)
    y = np.asarray(batch_n_values, dtype=np.float32)
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return None

    x_plot = x
    y_plot = y

    if x_plot.size == 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="#111")
        ax.set_facecolor("#111")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.5, float(MAX_ROLLOUT_LEN))
        ax.set_xlabel("Batch / Total Batches", color="white")
        ax.set_ylabel("Curriculum frontier N", color="white")
        ax.set_title(f"Epoch {epoch} Curriculum Frontier N", color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.text(
            0.5,
            0.5,
            "No curriculum points were collected for this epoch",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#cccccc",
            fontsize=10,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"curriculum_n_progress_ema_epoch_{epoch:03d}.png")
        plt.savefig(plot_path, dpi=150, facecolor="#111")
        plt.close()
        print(f"  Saved: {plot_path}")
        return plot_path

    ema = np.empty_like(y_plot)
    ema_state = None
    for i, v in enumerate(y_plot):
        ema_state = float(v) if ema_state is None else (float(ema_alpha) * float(v) + (1.0 - float(ema_alpha)) * ema_state)
        ema[i] = ema_state

    valid_points = int(x_plot.size)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), facecolor="#111")
    ax.set_facecolor("#111")
    ax.plot(x_plot, y_plot, color="#7bdff2", alpha=0.35, linewidth=1.0, label="Batch frontier N")
    ax.plot(x_plot, ema, color="#ffaa00", linewidth=2.0, label=f"EMA (alpha={ema_alpha:.2f})")
    ax.set_title(f"Epoch {epoch} Curriculum Frontier N", color="white", fontsize=12)
    ax.set_xlabel("Batch / Total Batches", color="white")
    ax.set_ylabel("Curriculum frontier N", color="white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.5, max(float(MAX_ROLLOUT_LEN), float(np.nanmax(ema) + 1.0 if np.isfinite(ema).any() else MAX_ROLLOUT_LEN)))
    ax.grid(color="#333", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"points: {valid_points}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#cccccc",
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"curriculum_n_progress_ema_epoch_{epoch:03d}.png")
    plt.savefig(plot_path, dpi=150, facecolor="#111")
    plt.close()

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch} curriculum N-progress EMA plot: {plot_path} "
                f"(alpha={ema_alpha:.2f}, points={valid_points})\n"
            )

    print(f"  Saved: {plot_path}")
    return plot_path

def save_rollout_video(model_to_render, sim_info, mean, std, zero_norm, get_labels_fn, out_path, title_tag):
    video_states = load_packed_array(sim_info["states_path"])
    warmup_idx = warmup_start_index(sim_info["n_frames"])
    usable_len = max(1, sim_info["n_frames"] - warmup_idx)
    start_idx = warmup_idx + int(usable_len * VIDEO_START_FRAC)
    end_idx = warmup_idx + int(usable_len * VIDEO_END_FRAC)
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
        writer = animation.FFMpegWriter(fps=FPS_VID, codec=VIDEO_CODEC, bitrate=VIDEO_BITRATE)
        writer_mode = "mp4"
    except (RuntimeError, FileNotFoundError, OSError):
        writer = animation.PillowWriter(fps=FPS_VID)

    def render_frame(idx, fig):
        fig.clf()
        gs = gridspec.GridSpec(2, 3, figure=fig, left=0.05, right=0.95, top=0.88, bottom=0.08, hspace=0.45, wspace=0.10)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        err_mag = vel_mag(gt_norm_seq[idx]) - vel_mag(pred_frames[idx])
        err_pres = pres(gt_norm_seq[idx]) - pres(pred_frames[idx])

        err_mag_abs = np.percentile(np.abs(err_mag), 100) if np.any(err_mag) else 1.0
        err_pres_abs = np.percentile(np.abs(err_pres), 100) if np.any(err_pres) else 1.0

        norm_err_mag = Normalize(vmin=-err_mag_abs, vmax=err_mag_abs)
        norm_err_pres = Normalize(vmin=-err_pres_abs, vmax=err_pres_abs)

        titles = ["GT  |u|", "Pred  |u|", "Error |u|", "GT  p", "Pred  p", "Error  p"]
        data = [
            vel_mag(gt_norm_seq[idx]), vel_mag(pred_frames[idx]), err_mag,
            pres(gt_norm_seq[idx]),    pres(pred_frames[idx]),    err_pres,
        ]
        cmaps = [cmap_vel, cmap_vel, "RdBu_r", cmap_pres, cmap_pres, "RdBu_r"]
        norms = [norm_vel, norm_vel, norm_err_mag, norm_pres, norm_pres, norm_err_pres]

        for ax, image, title, cmap, norm in zip(axes, data, titles, cmaps, norms):
            ax.set_facecolor("#111")
            im = ax.imshow(np.rot90(image, k=1), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=3)
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, format="%.3e")
            cb.ax.tick_params(colors="#ccc", labelsize=6)
            cb.outline.set_edgecolor("#555")

        fig.text(
            0.5, 0.94,
            f"{title_tag}   |   t = {start_idx + idx:05d}   |   Frame {idx:03d}/{len(pred_frames) - 1}",
            ha="center", va="top", color="white", fontsize=9, fontfamily="monospace",
        )

    fig = plt.figure(figsize=(18, 5.5), dpi=DPI_VID, facecolor="#111")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    setup_distributed()
    torch.manual_seed(GLOBAL_SEED + RANK)
    np.random.seed(GLOBAL_SEED + RANK)

    os.makedirs(OUT_DIR, exist_ok=True)
    print0(f"Device: {DEVICE}")
    print0(f"Output: {OUT_DIR}/")
    print0(f"DDP: enabled={DDP_ENABLED} world_size={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK}")

    print0("\nDiscovering simulation folders ...")
    sim_infos = discover_simulations(SIM_ROOT)
    print0(f"Found {len(sim_infos)} simulations under {SIM_ROOT}")
    total_frames = sum(sim["n_frames"] for sim in sim_infos)
    print0(f"Total frames across all simulations: {total_frames}")

    print0("\nEnsuring packed simulation caches ...")
    if is_main_process():
        ensure_all_sim_caches(sim_infos, CACHE_WORKERS, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)
    ddp_barrier()

    prepare_sim_cache_metadata(sim_infos)

    train_sim_infos, val_sim_infos = split_simulations(sim_infos, VAL_FRAC)
    print0(f"Train simulations: {len(train_sim_infos)}   Val simulations: {len(val_sim_infos)}")

    print0("Computing normalization statistics ...")
    mean, std = compute_global_stats(train_sim_infos, sim_infos)
    print0(f"  per-channel mean: {mean.numpy().round(5)}")
    print0(f"  per-channel std:  {std.numpy().round(5)}")

    train_ds = MultiSimKarmanDataset(train_sim_infos, mean, std)
    val_ds = MultiSimKarmanDataset(val_sim_infos, mean, std) if val_sim_infos else None

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
        if DDP_ENABLED
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=WORLD_SIZE, rank=RANK, shuffle=False)
        if (DDP_ENABLED and val_ds is not None)
        else None
    )

    train_loader = maybe_wrap_prefetch(build_loader(train_ds, shuffle=True, sampler=train_sampler))
    val_loader = maybe_wrap_prefetch(build_loader(val_ds, shuffle=False, sampler=val_sampler)) if val_ds is not None else None
    print0(f"Train samples: {len(train_ds)}   Val samples: {len(val_ds) if val_ds is not None else 0}")

    zero_norm = ((torch.zeros(3, device=DEVICE) - mean.to(DEVICE)) / std.to(DEVICE)).view(1, 3, 1, 1)
    if USE_CHANNELS_LAST:
        zero_norm = zero_norm.contiguous(memory_format=torch.channels_last)

    print0("\nBuilding PDE-Transformer (MC-S) ...")
    model = get_model()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print0(f"  Total parameters : {n_params:.2e} M")
    print0(f"  Trainable params : {n_trainable:.2e} M  (LoRA adapters only)")
    if DDP_ENABLED:
        ddp_kwargs = {
            "device_ids": [LOCAL_RANK] if DEVICE == "cuda" else None,
            "output_device": LOCAL_RANK if DEVICE == "cuda" else None,
            "find_unused_parameters": False,
        }
        try:
            model = DDP(
                model,
                static_graph=DDP_USE_STATIC_GRAPH,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                bucket_cap_mb=DDP_BUCKET_CAP_MB,
                **ddp_kwargs,
            )
            print0(
                f"  DDP fast-path: static_graph={DDP_USE_STATIC_GRAPH}, "
                "gradient_as_bucket_view=True, broadcast_buffers=False, "
                f"bucket_cap_mb={DDP_BUCKET_CAP_MB}"
            )
        except TypeError:
            model = DDP(model, **ddp_kwargs)
            print0("  DDP fast-path flags unavailable; using baseline DDP arguments.")
    if DEVICE == "cuda":
        print0("  Device prefetch: enabled")

    optimizer = create_optimizer(model)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHED_STEP_SIZE, gamma=SCHED_GAMMA)

    task_label = torch.tensor([TASK_LABEL_ID], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    train_losses = {"mse": []}
    val_losses = {"mse": []}
    best_val = math.inf
    loss_log_path = None
    start_epoch = 1
    current_promotion_improvement_frac = float(VELOCITY_IMPROVEMENT_FRAC)

    resume_path = os.path.join(OUT_DIR, "last.ckpt")
    if os.path.exists(resume_path):
        checkpoint_info = load_checkpoint(model, optimizer, scheduler, resume_path)
        resumed = checkpoint_info["resumed"]
        saved_epoch = checkpoint_info["saved_epoch"]
        best_val = checkpoint_info["best_val"]
        train_losses = checkpoint_info.get("train_losses", {"mse": []})
        val_losses   = checkpoint_info.get("val_losses",   {"mse": []})
        current_promotion_improvement_frac = float(
            checkpoint_info.get("promotion_improvement_frac_next", VELOCITY_IMPROVEMENT_FRAC)
        )
        print0(f"Loaded PEFT checkpoint: {resume_path}")
        if resumed:
            start_epoch = saved_epoch + 1
            print0(f"Successfully resumed training from epoch {saved_epoch}. Starting at epoch {start_epoch}.")
        else:
            print0("Loaded PEFT weights; optimizer/scheduler state not found.")

    if DDP_ENABLED:
        # Keep all ranks in sync with rank-0 resume metadata.
        meta = torch.tensor(
            [float(start_epoch), float(best_val), float(current_promotion_improvement_frac)],
            device=DEVICE,
        )
        dist.broadcast(meta, src=0)
        start_epoch = int(meta[0].item())
        best_val = float(meta[1].item())
        current_promotion_improvement_frac = float(meta[2].item())

    print0("\nWarm-up pass (lazy buffer initialisation) ...")
    model.eval()
    with torch.no_grad():
        _dummy_in = torch.zeros(
            WARMUP_DUMMY_BATCH,
            MODEL_IN_CHANNELS,
            MODEL_SAMPLE_SIZE,
            MODEL_SAMPLE_SIZE,
            device=DEVICE,
        )
        _ = model(_dummy_in, class_labels=get_labels(WARMUP_DUMMY_BATCH))
        del _dummy_in, _
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print0("  Done.")

    if not SKIP_TRAIN:
        if is_main_process():
            loss_log_path = os.path.join(OUT_DIR, f"loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(loss_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(
                    "timestamp epoch lr "
                    "train_mse avg_target_N train_frontier_vel frontier_baseline_vel "
                    "stage_batch_count promotion_count promotion_improvement_frac "
                    "promotion_improvement_frac_next max_frontier_N val_mse best\n"
                )

        for epoch in range(start_epoch, EPOCHS + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            if DDP_ENABLED:
                ddp_barrier()

            current_lr = optimizer.param_groups[0]["lr"]
            schedule_epoch = epoch - start_epoch + 1
            schedule_total_epochs = max(1, EPOCHS - start_epoch + 1)
            required_improvement = float(current_promotion_improvement_frac)
            print0(
                f"\nEpoch {epoch:02d}/{EPOCHS}  LR: {current_lr:.2e}  curriculum starts at N=1  "
                f"promo_impr={100.0 * required_improvement:.2f}%"
            )

            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                zero_norm=zero_norm,
                get_labels_fn=get_labels,
                training=True,
                optimizer=optimizer,
                epoch=schedule_epoch,
                total_epochs=schedule_total_epochs,
                promotion_improvement_frac=required_improvement,
            )

            if val_loader is None:
                val_loss = {"mse": float("nan"), "all_val_mses": None}
            else:
                val_loss = run_epoch(
                    model=model,
                    loader=val_loader,
                    zero_norm=zero_norm,
                    get_labels_fn=get_labels,
                    training=False,
                    epoch=schedule_epoch,
                    total_epochs=schedule_total_epochs,
                    promotion_improvement_frac=None,
                )
                if is_main_process() and val_loss["all_val_mses"] is not None:
                    # Pass the loss_log_path to write the table to it
                    analyze_and_plot(val_loss["all_val_mses"], OUT_DIR, epoch, log_path=loss_log_path)

            if is_main_process() and train_loss.get("batch_progress") is not None and train_loss.get("batch_n_values") is not None:
                plot_epoch_curriculum_n_progress_ema(
                    train_loss["batch_progress"],
                    train_loss["batch_n_values"],
                    OUT_DIR,
                    epoch,
                    ema_alpha=N_PROGRESS_EMA_ALPHA,
                    log_path=loss_log_path,
                )

            scheduler.step()
            train_losses["mse"].append(train_loss["mse"])
            val_losses["mse"].append(val_loss["mse"])

            if val_loss["mse"] < best_val:
                best_val = val_loss["mse"]
                best_marker = " ← best mse"
            else:
                best_marker = ""

            if train_loss["max_frontier_N"] < (MAX_ROLLOUT_LEN - 1):
                next_promotion_improvement_frac = max(
                    MIN_IMPROVEMENT_FRAC,
                    0.5 * float(current_promotion_improvement_frac),
                )
            else:
                next_promotion_improvement_frac = float(current_promotion_improvement_frac)

            if is_main_process():
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val,
                    train_losses,
                    val_losses,
                    next_promotion_improvement_frac,
                )

            current_promotion_improvement_frac = float(next_promotion_improvement_frac)

            if is_main_process() and epoch % 2 == 0:
                epoch_video_path = os.path.join(OUT_DIR, f"pred_vs_gt_epoch_{epoch:03d}.mp4")
                video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
                save_rollout_video(model, video_sim, mean, std, zero_norm, get_labels, epoch_video_path, f"Epoch {epoch:03d}")

            if DEVICE == "cuda":
                mem_line = (
                    f"  cuda mem: {torch.cuda.memory_allocated() / 1e9:.2e}GB / "
                    f"max {torch.cuda.max_memory_allocated() / 1e9:.2e}GB"
                )
            else:
                mem_line = "  cuda mem: n/a (cpu)"

            baseline_val = train_loss["frontier_baseline_vel"]
            baseline_str = f"{baseline_val:.6e}" if baseline_val is not None else "N/A"
            stage_frontier_vel_val = train_loss.get("running_stage_frontier_vel")
            if stage_frontier_vel_val is None or not math.isfinite(float(stage_frontier_vel_val)):
                stage_frontier_vel_str = "N/A"
            else:
                stage_frontier_vel_str = f"{float(stage_frontier_vel_val):.6e}"

            print0(
                f"  train: mse={train_loss['mse']:.6e}\n"
                f"         avg target N={train_loss['avg_target_N']:.2e}  current frontier N={train_loss['train_frontier_N']}  max frontier N={train_loss['max_frontier_N']}\n"
                f"         stage frontier vel={stage_frontier_vel_str}  epoch frontier vel={train_loss['train_frontier_vel']:.6e}  baseline vel={baseline_str}  stage batches={train_loss['stage_batch_count']}  promotions={train_loss['promotion_count']}\n"
                f"         required improvement(this epoch)={100.0 * train_loss['promotion_improvement_frac']:.2f}%  next epoch={100.0 * current_promotion_improvement_frac:.2f}% (last-{RECENT_PROMO_WINDOW} vs prior baseline)\n"
                f"  val  : mse={val_loss['mse']:.6e}{best_marker}\n"
                f"{mem_line}"
            )

            epoch_log_note = "best" if best_marker else ""
            if is_main_process() and loss_log_path:
                baseline_log_val = train_loss["frontier_baseline_vel"]
                baseline_log_str = f"{baseline_log_val:.6e}" if baseline_log_val is not None else "N/A"
                with open(loss_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{datetime.now().isoformat()} epoch={epoch} lr={current_lr:.2e} "
                        f"train_mse={train_loss['mse']:.6e} avg_target_N={train_loss['avg_target_N']:.2e} "
                        f"train_frontier_vel={train_loss['train_frontier_vel']:.6e} "
                        f"frontier_baseline_vel={baseline_log_str} "
                        f"stage_batch_count={train_loss['stage_batch_count']} "
                        f"promotion_count={train_loss['promotion_count']} "
                        f"promotion_improvement_frac={train_loss['promotion_improvement_frac']:.6e} "
                        f"promotion_improvement_frac_next={current_promotion_improvement_frac:.6e} "
                        f"max_frontier_N={train_loss['max_frontier_N']} "
                        f"val_mse={val_loss['mse']:.6e} {epoch_log_note}\n"
                    )

        if is_main_process():
            print("\nSaving loss curves ...")
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor="#111")
            ax.set_facecolor("#111")
            ax.plot(range(1, len(train_losses["mse"]) + 1), train_losses["mse"], color="#00c8ff", linewidth=2, label="Train MSE")
            ax.plot(range(1, len(val_losses["mse"]) + 1), val_losses["mse"], color="#ffaa00", linewidth=2, label="Val MSE")
            ax.set_xlabel("Epoch", color="white")
            ax.set_ylabel("MSE Loss", color="white")
            ax.set_title("MSE Loss (Fine-Tune)", color="white", fontsize=11)
            ax.legend(framealpha=0.3)
            ax.tick_params(colors="white")
            ax.set_yscale("log")
            for spine in ax.spines.values():
                spine.set_edgecolor("#555")
            plt.tight_layout()
            curve_path = os.path.join(OUT_DIR, "loss_curves.png")
            plt.savefig(curve_path, dpi=150, facecolor="#111")
            plt.close()
            print(f"  Saved: {curve_path}")
    else:
        print0("\nSkipping training (SKIP_TRAIN=True).")

    if not is_main_process():
        cleanup_distributed()
        return

    print("\nGenerating prediction vs ground-truth rollout video ...")

    best_path = os.path.join(OUT_DIR, "last.ckpt")
    if not os.path.exists(best_path):
        print("No last.ckpt found, skipping video.")
        return

    checkpoint = torch.load(best_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
    else:
        unwrap_model(model).load_state_dict(checkpoint)

    video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
    frame_path = save_rollout_video(
        model, video_sim, mean, std, zero_norm, get_labels,
        os.path.join(OUT_DIR, "pred_vs_gt.mp4"), "Final"
    )

    if not SKIP_TRAIN:
        print(f"\n{'=' * 80}")
        print("Fine-tuning complete.")
        print(f"  Best val mse   : {best_val:.6e}")
        if loss_log_path:
            print(f"  Loss log       : {loss_log_path}")
        print(f"  Loss curves    : {OUT_DIR}/loss_curves.png")
        print(f"  Prediction     : {frame_path}")
        print(f"  Checkpoint     : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 80}\n")
        header = f"{'Epoch':>6}  {'Train MSE':>11}  {'Val MSE':>11}"
        print(header)
        print("-" * len(header))
        for epoch_idx, (t_m, v_m) in enumerate(zip(train_losses["mse"], val_losses["mse"]), 1):
            print(f"{epoch_idx:>6}  {t_m:>11.6e}  {v_m:>11.6e}")
    else:
        print(f"\n{'=' * 80}")
        print("Visualization complete.")
        print(f"  Prediction   : {frame_path}")
        print(f"  Checkpoint   : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 80}\n")

    cleanup_distributed()


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)