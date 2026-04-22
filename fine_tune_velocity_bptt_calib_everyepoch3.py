"""
fine_tune_velocity_bptt_calib_everyepoch3.py
=====================
Fine-tune the frozen PDE-Transformer base model using LoRA (via PEFT) on
autoregressive multi-step rollout with Error-Velocity Truncation (EVT)
for long-rollout stability.

Key design decisions:
  - MSE loss is computed on raw model output (no mask in loss).
  - The mask is applied ONLY when advancing the rollout state.
  - EVT trigger: only fire if the step-wise velocity (E_t - E_{t-1})
    exceeds the calibrated threshold.
  - Truncated Pushforward: Instead of just backpropagating through a single 
    failed step, we backpropagate through a sliding window of previous steps 
    (BPTT_WINDOW) to teach the model how to prevent the error from compounding.
    - Deterministic EVT threshold (VEL_EPSILON) is calibrated from mean
        rollout velocity statistics; VEL_STD is retained for monitoring only.
  - LoRA (r=16, alpha=16) is injected into qkv, to_qkv, fc1, fc2 via PEFT.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import random
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
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Error-Velocity Truncation (EVT) Configuration
# ---------------------------------------------------------------------------
MAX_ROLLOUT_LEN = 12
BPTT_WINDOW = 3        # Number of steps to backpropagate through (Truncated Pushforward)

# Thresholds are set automatically by calibrate_velocity_threshold()
VEL_EPSILON   = None   # deterministic EVT threshold (mean velocity)
VEL_STD       = None   # diagnostic velocity spread (not used for threshold sampling)

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

torch.manual_seed(42)
np.random.seed(42)

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


def gather_triggered_ns(local_list):
    if not DDP_ENABLED:
        return local_list

    local_np = np.asarray(local_list, dtype=np.int64)
    local_len = int(local_np.size)

    len_tensor = torch.tensor([local_len], dtype=torch.int64, device=DEVICE)
    gathered_lens_t = [torch.zeros(1, dtype=torch.int64, device=DEVICE) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_lens_t, len_tensor)
    lengths = [int(t.item()) for t in gathered_lens_t]
    max_len = max(lengths) if lengths else 0

    padded_local = torch.zeros((max_len,), dtype=torch.int64, device=DEVICE)
    if local_len > 0:
        padded_local[:local_len] = torch.as_tensor(local_np, dtype=torch.int64, device=DEVICE)

    gathered = [torch.empty_like(padded_local) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered, padded_local)

    if not is_main_process():
        return None

    merged = []
    for rank_tensor, rank_len in zip(gathered, lengths):
        if rank_len > 0:
            merged.extend(rank_tensor[:rank_len].cpu().tolist())
    return merged

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------
SIM_ROOT = "./data/256_inc"
OUT_DIR = os.path.join("runs", "karman_finetuned_velocity_LoRA_bptt_calib_everyepoch3")
EPOCHS = 40
BATCH_SIZE = 12
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
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_CHANNELS_LAST = False
NUM_WORKERS = max(0, cpu_count() - 5)
PIN_MEMORY = DEVICE == "cuda"
CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
CACHE_WORKERS = max(1, cpu_count() - 5)
PREFETCH_FACTOR = 2
TQDM_UPDATE_EVERY = 20
DDP_BUCKET_CAP_MB = 100
DDP_USE_STATIC_GRAPH = False


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
    std = torch.tensor(stacked.std(axis=(0, 2, 3)), dtype=torch.float32) + 1e-6
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
        # Cap per-rank loader workers so torchrun does not spawn an excessive
        # number of processes across all ranks.
        per_rank_workers = max(0, min(NUM_WORKERS, max(1, cpu_count() // max(1, WORLD_SIZE)) - 1))
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
        sample_size=256,
        in_channels=3,
        out_channels=3,
        type=MODEL_TYPE,
        patch_size=4,
        periodic=False,
        carrier_token_active=True,
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
        r=16,
        lora_alpha=16,
        target_modules=["qkv", "to_qkv", "fc1", "fc2"],
        lora_dropout=0.05,
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
        "weight_decay": 1e-6,
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


def update_progress(progress_bar, loss_dict, sample_count):
    denom = max(1, sample_count)
    avg = {}
    for key, value in loss_dict.items():
        v = value / denom
        avg[key] = v.detach().item() if isinstance(v, torch.Tensor) else float(v)
    progress_bar.set_postfix(
        mse=f"{avg['mse']:.6e}",
        N=f"{avg.get('N', 0):.2e}",
        vel=f"{avg.get('vel', 0):.6e}",
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    base_model = unwrap_model(model)
    return {
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val": best_metric,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def normalize_loss_history(losses):
    if not isinstance(losses, dict):
        return {"mse": []}
    return {"mse": list(losses.get("mse", []))}


def per_sample_mse(pred, target):
    return F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))


def positive_velocity_stat(curr_err_per_sample, prev_err_per_sample):
    vel = curr_err_per_sample - prev_err_per_sample
    pos_vel = vel[vel > 0]
    if pos_vel.numel() == 0:
        return None, 0
    return float(pos_vel.mean().item()), int(pos_vel.numel())


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    checkpoint = build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses)
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
        }

    return {
        "resumed": False,
        "saved_epoch": None,
        "best_val": math.inf,
        "train_losses": {"mse": []},
        "val_losses": {"mse": []},
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_velocity_threshold(model, loader, zero_norm, get_labels_fn):
    """Survey a fraction of the train loader *once* with no gradients to collect
    the first-differences of MSE errors over a rollout-step range.

    Velocity at step t:
        v_t = E_t - E_{t-1}

    We collect v_t for t in [calib_start_step, calib_end_step] across the first
    20% of training batches, then set a deterministic threshold from the mean.

    Each epoch uses the current calibration result.
    """
    global VEL_EPSILON, VEL_STD

    calib_start_step = max(1, int(MAX_ROLLOUT_LEN * 0.25))
    calib_end_step = max(calib_start_step, int(MAX_ROLLOUT_LEN * 0.75) - 1)
    min_steps_needed = calib_end_step + 1

    print0(
        f"\nCalibrating EVT thresholds from train set "
        f"(steps {calib_start_step}..{calib_end_step}) ..."
    )
    print0("  Using first 20% of train batches for speed.")
    model.eval()
    v_values = []
    max_calib_batches = max(1, len(loader) // 5)

    with torch.inference_mode():
        for batch_idx, (x_batch, y_seq, mask) in enumerate(
            tqdm(loader, desc="calib", leave=False, disable=not is_main_process(), dynamic_ncols=True, total=max_calib_batches)
        ):
            if batch_idx >= max_calib_batches:
                break

            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])

            if y_seq.shape[1] < min_steps_needed:
                continue

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            state = x_batch.clone()
            prev_err_per_sample = None

            for t in range(min_steps_needed):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(state, class_labels=labels).sample
                pred = pred.float()

                y_t = y_seq[:, t]
                mse_per_sample = per_sample_mse(pred, y_t)

                if not torch.isfinite(mse_per_sample).all():
                    prev_err_per_sample = None
                    break

                if prev_err_per_sample is not None and calib_start_step <= t <= calib_end_step:
                    step_stat, _ = positive_velocity_stat(mse_per_sample, prev_err_per_sample)
                    if step_stat is not None:
                        v_values.append(step_stat)

                prev_err_per_sample = mse_per_sample
                state = torch.lerp(zero_norm, pred, mask)

    local_count = float(len(v_values))
    local_sum = float(np.sum(v_values)) if v_values else 0.0
    local_sum_sq = float(np.sum(np.square(v_values))) if v_values else 0.0

    stats = torch.tensor([local_count, local_sum, local_sum_sq], dtype=torch.float64, device=DEVICE)
    if DDP_ENABLED:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_count, global_sum, global_sum_sq = stats.tolist()
    print0(f"  Positive velocity batch-step candidates: {int(global_count)}")

    if global_count <= 0:
        fallback_v_eps = 1e-4
        print0("  WARNING: no valid calibration samples found. Using fallback thresholds.")
        current_v_eps = fallback_v_eps
        current_v_std = 0.5 * fallback_v_eps
    else:
        current_v_eps = float(global_sum / global_count)
        current_v_std = float(np.sqrt(max((global_sum_sq / global_count) - ((global_sum / global_count) ** 2), 0.0)))

        if current_v_eps <= 0:
            fallback = np.array(v_values, dtype=np.float64)
            local_floor = float(np.percentile(fallback, 75)) if fallback.size > 0 else 1e-8
            floor_t = torch.tensor(local_floor, dtype=torch.float64, device=DEVICE)
            if DDP_ENABLED:
                dist.all_reduce(floor_t, op=dist.ReduceOp.MAX)
            current_v_eps = max(float(floor_t.item()), 1e-8)
            print0("  NOTE: mean velocity <= 0; using 75th-percentile as floor.")

    VEL_EPSILON = float(current_v_eps)
    VEL_STD = float(current_v_std)

    print0(f"  Calibrated VEL_EPSILON   = {VEL_EPSILON:.6e}")
    print0(f"  Calibrated VEL_STD       = {VEL_STD:.6e}")


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, zero_norm, get_labels_fn, training, optimizer=None, global_step=0):
    """Run one full epoch.

    MSE loss contract:
        loss = F.mse_loss(pred_raw, y_t)

    Masking (obstacle enforcement):
        state = torch.lerp(zero_norm, pred_raw, mask)

    Error-Velocity Truncation (training only):
        - At each step t >= 1, compute the first-difference:
              v_t = E_t - E_{t-1}
        - Trigger condition: v_t > vel_threshold
              where threshold ~ Normal(EPSILON, STD) per batch.
        - When triggered: Backpropagate through a sliding window (BPTT_WINDOW)
          ending at step t-1 using the Pushforward method.
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
        "N":     0,     
        "vel":   0.0,   
    }
    sample_count = 0
    all_val_mses = [] if not training else None
    triggered_ns = [] if training else None
    max_rollout_seen = 0
    max_target_N = -1
    missing_pos_vel_steps = 0
    vel_candidate_steps = 0

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    n_batches = len(loader)

    if training:
        # Keep stochastic controls identical across DDP ranks.
        rng_seed = 12345 + int(global_step)
        py_rng = random.Random(rng_seed)
        vel_threshold = max(0.0, float(VEL_EPSILON)) if VEL_EPSILON is not None else 2e-5
    else:
        vel_threshold = 2e-5

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        progress_bar = make_progress(loader, desc)
        for step, (x_batch, y_seq, mask) in enumerate(progress_bar):
            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])
            batch_size = x_batch.shape[0]

            max_rollout = min(MAX_ROLLOUT_LEN, y_seq.shape[1])
            max_rollout_seen = max(max_rollout_seen, max_rollout)

            current_vel_threshold = vel_threshold

            if mask.ndim == 3:   
                mask = mask.unsqueeze(1)   

            if training:
                states_seq = []  
                prev_err_per_sample = None
                pos_vel_sum = 0.0
                pos_vel_count = 0
                target_N = -1     

                rand_val = py_rng.random()
                if rand_val < 0.25:
                    fixed_target = 0
                else:
                    fixed_target = -1

                if fixed_target != -1 and fixed_target >= max_rollout:
                    fixed_target = max_rollout - 1

                with torch.no_grad():
                    state = x_batch
                    for t in range(max_rollout):
                        states_seq.append(state)

                        if fixed_target != -1 and t == fixed_target:
                            target_N = t
                            break

                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                            pred_dry = model(state, class_labels=labels).sample
                        pred_dry = pred_dry.float()

                        mse_per_sample = per_sample_mse(pred_dry, y_seq[:, t])
                        if not torch.isfinite(mse_per_sample).all():
                            break

                        target_N = t

                        if fixed_target == -1 and prev_err_per_sample is not None:
                            vel_candidate_steps += 1
                            curr_vel, _ = positive_velocity_stat(mse_per_sample, prev_err_per_sample)
                            if curr_vel is not None:
                                pos_vel_sum += curr_vel
                                pos_vel_count += 1
                                if curr_vel > current_vel_threshold:
                                    target_N = t - 1
                                    break
                            else:
                                missing_pos_vel_steps += 1

                        prev_err_per_sample = mse_per_sample
                        state = torch.lerp(zero_norm, pred_dry, mask)

                local_available_last = len(states_seq) - 1

                # Keep truncation decision local per rank so EVT semantics
                # reflect the local positive-velocity statistic.
                if target_N >= 0:
                    target_N = min(target_N, local_available_last)

                if target_N < 0:
                    target_N = -1

                if triggered_ns is not None:
                    triggered_ns.append(int(target_N))

                # --- Phase 2: Sliding Window Pushforward with Grad ---
                loss_to_backprop = None
                
                if target_N >= 0:
                    # Calculate where our gradient window should start
                    window_start = max(0, target_N - BPTT_WINDOW + 1)
                    
                    # Grab the clean, detached input state from Phase 1
                    current_state = states_seq[window_start].detach()
                    window_loss = torch.zeros((), device=DEVICE)
                    weight_sum = 0.0
                    
                    for t in range(window_start, target_N + 1):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                            pred_raw = model(current_state, class_labels=labels).sample
                        pred_raw = pred_raw.float()
                        
                        # Geometric Weighting: Target step gets full weight. Steps leading up to it get partial weight.
                        weight = 1.0 if t == target_N else 0.5
                        
                        step_loss = F.mse_loss(pred_raw, y_seq[:, t])
                        window_loss += step_loss * weight
                        weight_sum += weight
                        
                        # If this isn't the last step in the window, advance the state differentiably
                        if t < target_N:
                            current_state = torch.lerp(zero_norm, pred_raw, mask)
                            
                    # Average the accumulated loss over the window size
                    loss_to_backprop = window_loss / max(weight_sum, 1e-12)

                if loss_to_backprop is None:
                    # Keep DDP ranks synchronized: even if EVT finds no valid
                    # rollback target on this rank, run a zero-weight loss so
                    # every rank still executes backward/all-reduce.
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                        pred_fallback = model(x_batch, class_labels=labels).sample
                    pred_fallback = pred_fallback.float()
                    loss_to_backprop = F.mse_loss(pred_fallback, y_seq[:, 0]) * 0.0

                (loss_to_backprop / ACCUM_GRAD).backward()

                if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == n_batches:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if target_N >= 0:
                    loss_accum["mse"] += loss_to_backprop.detach() * batch_size
                    loss_accum["N"] += target_N * batch_size
                    max_target_N = max(max_target_N, target_N)
                    avg_v = pos_vel_sum / pos_vel_count if pos_vel_count > 0 else 0.0
                    loss_accum["vel"] += avg_v * batch_size
                    sample_count += batch_size

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
                update_progress(progress_bar, loss_accum, sample_count)

    loss_tensor = torch.tensor(
        [
            float(loss_accum["mse"].detach().item()),
            float(loss_accum["N"]),
            float(loss_accum["vel"]),
            float(sample_count),
        ],
        device=DEVICE,
        dtype=torch.float64,
    )
    if DDP_ENABLED:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

    max_target_tensor = torch.tensor([float(max_target_N)], device=DEVICE, dtype=torch.float64)
    if DDP_ENABLED:
        dist.all_reduce(max_target_tensor, op=dist.ReduceOp.MAX)

    vel_stat_tensor = torch.tensor(
        [float(missing_pos_vel_steps), float(vel_candidate_steps)],
        device=DEVICE,
        dtype=torch.float64,
    )
    if DDP_ENABLED:
        dist.all_reduce(vel_stat_tensor, op=dist.ReduceOp.SUM)

    total_mse, total_N, total_vel, total_samples = loss_tensor.tolist()
    total_missing_pos_steps, total_candidate_steps = vel_stat_tensor.tolist()
    denom = max(1.0, total_samples)
    max_N_reached = int(max_target_tensor.item())
    pos_vel_none_pct = (100.0 * total_missing_pos_steps / total_candidate_steps) if total_candidate_steps > 0 else 0.0

    if not training:
        local_val = (
            np.concatenate(all_val_mses, axis=0)
            if all_val_mses
            else np.empty((0, max_rollout_seen), dtype=np.float32)
        )
        gathered_val = gather_val_mses(local_val)
    else:
        gathered_val = None

    gathered_triggered_ns = gather_triggered_ns(triggered_ns) if training else None

    return {
        "mse": total_mse / denom,
        "N": total_N / denom if training else 0.0,
        "vel": total_vel / denom if training else 0.0,
        "max_N": max_N_reached if training else max_rollout_seen - 1,
        "pos_vel_none_pct": pos_vel_none_pct if training else 0.0,
        "all_val_mses": gathered_val if (not training and is_main_process()) else None,
        "triggered_ns": gathered_triggered_ns if (training and is_main_process()) else None,
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


def plot_triggered_n_distribution(triggered_ns, out_dir, epoch, log_path=None):
    os.makedirs(out_dir, exist_ok=True)

    values = np.asarray(triggered_ns, dtype=np.int64)
    max_bin = MAX_ROLLOUT_LEN - 1
    if values.size > 0:
        max_bin = max(max_bin, int(values.max()))

    counts = np.bincount(values, minlength=max_bin + 1) if values.size > 0 else np.zeros(max_bin + 1, dtype=np.int64)
    steps = np.arange(counts.shape[0])

    summary_lines = [
        "",
        f"--- Epoch {epoch} Triggered N Distribution ---",
        f"Total batches: {int(values.size)}",
    ]
    for step, count in zip(steps, counts):
        summary_lines.append(f"N={step:02d}: {int(count)}")
    summary_str = "\n".join(summary_lines)
    print(summary_str)

    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(summary_str + "\n\n")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), facecolor="#111")
    ax.set_facecolor("#111")
    ax.bar(steps, counts, color="#7bdff2", edgecolor="#d6f6ff", linewidth=0.8)
    ax.set_title("Triggered N Distribution per Epoch", color="white", fontsize=12)
    ax.set_xlabel("Triggered rollout step N", color="white")
    ax.set_ylabel("Batch count", color="white")
    ax.set_xticks(steps)
    ax.tick_params(colors="white")
    ax.text(
        0.5,
        0.98,
        "Includes fixed_target=0 batches",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="#cccccc",
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"triggered_n_distribution_epoch_{epoch:03d}.png")
    plt.savefig(plot_path, dpi=150, facecolor="#111")
    plt.close()

    print(f"  Saved: {plot_path}")

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
    torch.manual_seed(42 + RANK)
    np.random.seed(42 + RANK)

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    task_label = torch.tensor([1000], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    train_losses = {"mse": []}
    val_losses = {"mse": []}
    best_val = math.inf
    loss_log_path = None
    start_epoch = 1

    resume_path = os.path.join(OUT_DIR, "last.ckpt")
    if os.path.exists(resume_path):
        checkpoint_info = load_checkpoint(model, optimizer, scheduler, resume_path)
        resumed = checkpoint_info["resumed"]
        saved_epoch = checkpoint_info["saved_epoch"]
        best_val = checkpoint_info["best_val"]
        train_losses = checkpoint_info.get("train_losses", {"mse": []})
        val_losses   = checkpoint_info.get("val_losses",   {"mse": []})
        print0(f"Loaded PEFT checkpoint: {resume_path}")
        if resumed:
            start_epoch = saved_epoch + 1
            print0(f"Successfully resumed training from epoch {saved_epoch}. Starting at epoch {start_epoch}.")
        else:
            print0("Loaded PEFT weights; optimizer/scheduler state not found.")

    if DDP_ENABLED:
        # Keep all ranks in sync with rank-0 resume metadata.
        meta = torch.tensor([float(start_epoch), float(best_val)], device=DEVICE)
        dist.broadcast(meta, src=0)
        start_epoch = int(meta[0].item())
        best_val = float(meta[1].item())

    print0("\nWarm-up pass (lazy buffer initialisation) ...")
    model.eval()
    with torch.no_grad():
        _dummy_in = torch.zeros(1, 3, 256, 256, device=DEVICE)
        _ = model(_dummy_in, class_labels=get_labels(1))
        del _dummy_in, _
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print0("  Done.")

    if not SKIP_TRAIN:
        if is_main_process():
            loss_log_path = os.path.join(OUT_DIR, f"loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(loss_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(
                    "timestamp epoch lr bptt "
                    "train_mse train_N train_vel "
                    "val_mse best\n"
                )

        for epoch in range(start_epoch, EPOCHS + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            # Calibrate before every epoch so EVT thresholds track the model as it changes.
            calibrate_velocity_threshold(model, train_loader, zero_norm, get_labels)

            print0("\nPost-calibration warm-up (de-tainting inference tensors) ...")
            model.train()
            with torch.enable_grad():
                _dummy_in = torch.zeros(1, 3, 256, 256, device=DEVICE, requires_grad=False)
                _dummy_out = model(_dummy_in, class_labels=get_labels(1)).sample
                _dummy_out.sum().backward()
                del _dummy_in, _dummy_out
            optimizer.zero_grad(set_to_none=True)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            print0("  Done.")

            if DDP_ENABLED:
                ddp_barrier()

            current_lr = optimizer.param_groups[0]["lr"]
            print0(f"\nEpoch {epoch:02d}/{EPOCHS}  LR: {current_lr:.2e}  "
                   f"EVT_v_eps={VEL_EPSILON:.3e}  BPTT_W={BPTT_WINDOW}")

            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                zero_norm=zero_norm,
                get_labels_fn=get_labels,
                training=True,
                optimizer=optimizer,
                global_step=(epoch - 1) * len(train_loader),
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
                )
                if is_main_process() and val_loss["all_val_mses"] is not None:
                    # Pass the loss_log_path to write the table to it
                    analyze_and_plot(val_loss["all_val_mses"], OUT_DIR, epoch, log_path=loss_log_path)

            if is_main_process() and train_loss.get("triggered_ns") is not None:
                plot_triggered_n_distribution(train_loss["triggered_ns"], OUT_DIR, epoch, log_path=loss_log_path)

            scheduler.step()
            train_losses["mse"].append(train_loss["mse"])
            val_losses["mse"].append(val_loss["mse"])

            if val_loss["mse"] < best_val:
                best_val = val_loss["mse"]
                best_marker = " ← best mse"
            else:
                best_marker = ""

            if is_main_process():
                save_checkpoint(model, optimizer, scheduler, epoch, best_val, train_losses, val_losses)

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

            print0(
                f"  train: mse={train_loss['mse']:.6e}\n"
                f"         avg rollout N={train_loss['N']:.2e}  max N reached={train_loss['max_N']}  max allowed N={MAX_ROLLOUT_LEN}  avg_vel={train_loss['vel']:.6e}\n"
                f"         pos-vel none pct={train_loss['pos_vel_none_pct']:.2f}%\n"
                f"  val  : mse={val_loss['mse']:.6e}{best_marker}\n"
                f"{mem_line}"
            )

            epoch_log_note = "best" if best_marker else ""
            if is_main_process() and loss_log_path:
                with open(loss_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{datetime.now().isoformat()} epoch={epoch} lr={current_lr:.2e} bptt={BPTT_WINDOW} "
                        f"train_mse={train_loss['mse']:.6e} train_N={train_loss['N']:.2e} "
                        f"train_max_N={train_loss['max_N']} "
                        f"train_vel={train_loss['vel']:.6e} "
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