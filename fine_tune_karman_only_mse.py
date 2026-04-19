"""
fine_tune_karman_only_mse.py
============================
Fine-tune the frozen PDE-Transformer base model by training only the
injected SequentialModel sub-network. Uses autoregressive multi-step
rollout with Error-Acceleration Truncation (EAT) for long-rollout stability.

Key design decisions:
  - MSE loss is computed on raw model output (no mask in loss, same as train_karman_mse.py).
  - The mask is applied ONLY when advancing the rollout state (lerp to zero_norm inside obstacle).
  - EAT trigger: only fire if the *average* acceleration over two consecutive steps exceeds the
    calibrated threshold; backprop through the *first* of the two triggering steps.
  - Tolerance (ACCEL_EPSILON) and its per-sample noise std (ACCEL_STD) are calibrated
    automatically from a single no-grad survey pass over the train set before training begins,
    targeting the mean and std of the step-5 second-difference of MSE errors.
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
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import transformers
from pdetransformer.core.sub_network.llm import SequentialModel

# ---------------------------------------------------------------------------
# LLM Sub-Network Configuration
# ---------------------------------------------------------------------------
SEQ_HIDDEN_SIZE = 144
SEQ_NUM_HEADS = 8
SEQ_N_LAYERS = 6
# Options for SEQ_ATTN_METHOD:
#   "hyper"  – HyperAttention (requires Triton, Linux only)
#   "naive"  – vanilla PyTorch scaled_dot_product_attention (cross-platform)
SEQ_ATTN_METHOD = "naive"

# ---------------------------------------------------------------------------
# Error-Acceleration Truncation (EAT) Configuration
# ---------------------------------------------------------------------------
MAX_ROLLOUT_LEN = 15
# ACCEL_EPSILON and ACCEL_STD are set automatically by calibrate_accel_threshold()
# before training starts.  These are overwritten at runtime.
ACCEL_EPSILON = None   # mean acceleration at step 5 across train set
ACCEL_STD     = None   # std  acceleration at step 5 across train set

# Target rollout step at which calibration is measured (0-indexed step index).
# Step 4 means the 5th model call, which gives the first "second difference"
# that uses errors from steps 2,3,4  (a_t = E_t - 2*E_{t-1} + E_{t-2}).
CALIB_TARGET_STEP = 4  # 0-indexed → step 5 in 1-indexed language


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


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------
SIM_ROOT = "./256_inc"
OUT_DIR = os.path.join("runs", "karman_finetuned")
EPOCHS = 40
BATCH_SIZE = 48
ACCUM_GRAD = 1
LR = 4e-5
VAL_FRAC = 0.10
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
            # Ensure there is at least max_rollout ground-truth frames after x.
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
class LatentWrapper(nn.Module):
    def __init__(self, orig_latent, seq_model):
        super().__init__()
        self.orig_latent = orig_latent
        self.seq_model = seq_model
        # Learned importance weight for the sub-network path.
        # Initialized to 1e-6 for a stable start that still allows the gradient signal to flow.
        self.gamma = nn.Parameter(torch.full((1,), 1e-6))

    def forward(self, x, c):
        x = self.orig_latent(x, c)
        original_dtype = x.dtype
        x_5d = x.unsqueeze(-1)       # [N, C, H, W, 1]
        out_5d = self.seq_model(x_5d)
        out_4d = out_5d.squeeze(-1)
        # Apply the learned importance weight
        x = x + self.gamma * out_4d
        return x.to(original_dtype)


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
    ).to(DEVICE)

    if USE_CHANNELS_LAST:
        base_model = base_model.to(memory_format=torch.channels_last)

    for param in base_model.parameters():
        param.requires_grad = False

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
    ).to(DEVICE)

    if USE_CHANNELS_LAST:
        seq_model = seq_model.to(memory_format=torch.channels_last)

    # Compile the sub-network for faster repeated forward passes.
    # mode="reduce-overhead" gives most of the speedup with a short compile time.
    # Applied before LatentWrapper attachment so compile sees the standalone module.
    # Skipped on Windows: the inductor backend requires Triton which is Linux-only;
    # torch.compile() itself succeeds but the first forward pass raises TritonMissing.
    if torch.cuda.is_available() and os.name != "nt":
        try:
            seq_model = torch.compile(seq_model, mode="reduce-overhead")
        except Exception:
            pass  # silent fallback to eager

    base_model.model.latent = LatentWrapper(base_model.model.latent, seq_model)
    base_model.model.latent.to(DEVICE)
    return base_model


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
    if DEVICE != "cuda":
        x = x.to(DEVICE, non_blocking=PIN_MEMORY)
        y_seq = y_seq.to(DEVICE, non_blocking=PIN_MEMORY)
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
        mse=f"{avg['mse']:.6f}",
        N=f"{avg.get('N', 0):.2f}",
        accel=f"{avg.get('accel', 0):.6f}",
        imp=f"{avg.get('importance', 0):.6f}",
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
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


def normalize_loss_history(losses):
    if not isinstance(losses, dict):
        return {"mse": []}
    return {"mse": list(losses.get("mse", []))}


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    checkpoint = build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses)
    torch.save(checkpoint, os.path.join(OUT_DIR, "last.ckpt"))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Remap keys: old plain latent → new LatentWrapper.orig_latent
    new_state_dict = {}
    for k, v in state_dict.items():
        if (k.startswith("model.latent.")
                and not k.startswith("model.latent.orig_latent.")
                and not k.startswith("model.latent.seq_model.")):
            new_k = k.replace("model.latent.", "model.latent.orig_latent.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Intentionally do NOT restore optimizer/scheduler — fresh fine-tune.
        return {
            "resumed": True,
            "saved_epoch": int(checkpoint.get("epoch", 0)),
            "best_val": float(checkpoint.get("best_val", math.inf)),
            "train_losses": {"mse": []},
            "val_losses": {"mse": []},
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
def calibrate_accel_threshold(model, loader, zero_norm, get_labels_fn):
    """Survey a fraction of the train loader *once* with no gradients to collect
    the *relative* second-difference of MSE errors at the target step.

    Relative (dimensionless) acceleration at step t:

        a_rel_t = (E_t - 2*E_{t-1} + E_{t-2}) / mean(E_{t-2}, E_{t-1}, E_t)

    The denominator is the mean of the three-point window.  This makes
    a_rel_t a pure curvature-fraction — independent of the model's current
    absolute error level — so the calibrated threshold stays valid for the
    entire training run, even as MSE drops by orders of magnitude.

    We collect a_rel_t at CALIB_TARGET_STEP across the first 10% of training
    batches, then set:
        ACCEL_EPSILON = mean(a_rel values)
        ACCEL_STD     = std(a_rel values)
    """
    global ACCEL_EPSILON, ACCEL_STD

    min_steps_needed = CALIB_TARGET_STEP + 1   # need steps 0..CALIB_TARGET_STEP
    if min_steps_needed < 3:
        raise ValueError("CALIB_TARGET_STEP must be >= 2 so that a second-difference exists.")

    print(f"\nCalibrating EAT threshold from train set (target step {CALIB_TARGET_STEP + 1}) ...")
    print(f"  Using first 10% of train batches for speed.")
    print(f"  Threshold is RELATIVE (dimensionless curvature fraction).")
    model.eval()
    a_values = []
    max_calib_batches = max(1, len(loader) // 10)

    with torch.inference_mode():
        for batch_idx, (x_batch, y_seq, mask) in enumerate(
            tqdm(loader, desc="calib", leave=False, dynamic_ncols=True, total=max_calib_batches)
        ):
            if batch_idx >= max_calib_batches:
                break

            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])

            if y_seq.shape[1] < min_steps_needed:
                continue

            # Normalise mask to [B, 1, H, W] for lerp broadcasting
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            state = x_batch.clone()
            errors = []

            for t in range(min_steps_needed):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(state, class_labels=labels).sample
                pred = pred.float()

                y_t = y_seq[:, t]
                mse_val = F.mse_loss(pred, y_t).item()

                if not math.isfinite(mse_val):
                    errors = []
                    break

                errors.append(mse_val)
                state = torch.lerp(zero_norm, pred, mask)

            if len(errors) == min_steps_needed:
                e0, e1, e2 = errors[-3], errors[-2], errors[-1]
                window_mean = (e0 + e1 + e2) / 3.0
                if window_mean > 1e-12:   # guard against pathological near-zero errors
                    a_rel = (e2 - 2.0 * e1 + e0) / window_mean
                    a_values.append(a_rel)

    if not a_values:
        fallback_eps = 0.05   # dimensionless fallback: 5% curvature fraction
        print(f"  WARNING: no valid calibration samples found. Using fallback ACCEL_EPSILON={fallback_eps}")
        ACCEL_EPSILON = fallback_eps
        ACCEL_STD = 0.5 * fallback_eps
    else:
        arr = np.array(a_values, dtype=np.float64)
        ACCEL_EPSILON = float(np.mean(arr))
        ACCEL_STD = float(np.std(arr))
        # Guard: if mean is near zero or negative, use a small positive floor
        if ACCEL_EPSILON <= 0:
            ACCEL_EPSILON = max(float(np.percentile(arr, 75)), 1e-4)
            print(f"  NOTE: mean relative acceleration <= 0; using 75th-percentile as floor.")


    print(f"  Calibrated ACCEL_EPSILON = {ACCEL_EPSILON:.6e}")
    print(f"  Calibrated ACCEL_STD     = {ACCEL_STD:.6e}")


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------
def run_epoch(model, loader, zero_norm, get_labels_fn, training, optimizer=None, global_step=0):
    """Run one full epoch.

    MSE loss contract (matching train_karman_mse.py):
        loss = F.mse_loss(pred_raw, y_t)   # no mask applied before the loss

    Masking (obstacle enforcement):
        state = torch.lerp(zero_norm, pred_raw, mask)  # ONLY for the rollout state

    Error-Acceleration Truncation (training only):
        - At each step t >= 2, compute the second-difference:
              a_t = E_t - 2*E_{t-1} + E_{t-2}
        - We record (a_{t-1}, a_t) as consecutive pairs.
        - Trigger condition: mean(a_{t-1}, a_t) > threshold
              where threshold ~ Normal(ACCEL_EPSILON, ACCEL_STD) per batch.
        - When triggered: backprop through the *first* of the two steps (step t-1),
          i.e. use loss_at_{t-1} for the backward pass.
        - This prevents reacting to momentary spikes and ensures the gradient
          signal comes from the first step where drift begins.
    """
    if training:
        # Base model frozen → keep it in eval mode. Only seq_model trains.
        model.eval()
        model.model.latent.seq_model.train()
        optimizer.zero_grad(set_to_none=True)
        desc = "tr"
    else:
        model.eval()
        desc = "va"

    loss_accum = {
        "mse":   torch.zeros((), device=DEVICE),
        "N":     0,     # plain Python int  — avoids GPU tensor alloc + sync overhead
        "accel": 0.0,   # plain Python float — same reason
        "importance": torch.zeros((), device=DEVICE),
    }
    sample_count = 0

    # Cache once — avoids traversing the full parameter tree on every optimizer step.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Pre-generate all per-batch EAT thresholds for the epoch at once (outside the hot loop)
    # so no NumPy calls happen inside the batch iteration.
    n_batches = len(loader)
    if training and ACCEL_EPSILON is not None and ACCEL_STD is not None:
        _thresholds = np.maximum(
            0.0, np.random.normal(loc=ACCEL_EPSILON, scale=ACCEL_STD, size=n_batches)
        ).tolist()  # plain Python floats — no NumPy inside the loop
    else:
        _thresholds = [2e-5] * n_batches  # fallback; also used for val (unused there)

    # Mirror train_karman_mse.py: one outer context wrapping the entire loop.
    # torch.enable_grad() is critical after calibrate_accel_threshold() ran under
    # torch.no_grad(), to ensure the autograd engine is fully re-enabled.
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        progress_bar = make_progress(loader, desc)
        for step, (x_batch, y_seq, mask) in enumerate(progress_bar):
            x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
            labels = get_labels_fn(x_batch.shape[0])
            batch_size = x_batch.shape[0]

            max_rollout = min(MAX_ROLLOUT_LEN, y_seq.shape[1])

            # Index into the pre-generated list — zero NumPy overhead per batch
            current_threshold = _thresholds[step]

            # Ensure mask is [B, 1, H, W] so it broadcasts correctly to [B, C, H, W] in lerp.
            # The DataLoader stacks per-sample masks: [H,W] → [B,H,W] or [1,H,W] → [B,1,H,W].
            # We normalise to 4D here once, avoiding repeated unsq inside the rollout.
            if mask.ndim == 3:   # [B, H, W]
                mask = mask.unsqueeze(1)   # → [B, 1, H, W]

            if training:
                # ----------------------------------------------------------
                # Training: two-phase EAT rollout.
                #
                # Phase 1 – dry-run (torch.no_grad): advance the rollout and
                #   determine which step t* to train on, recording the *input
                #   state* at every step.  No autograd graphs are built, so
                #   memory cost is one [B,C,H,W] tensor per step instead of
                #   one full backward graph per step.
                #
                # Phase 2 – re-materialise only step t* with grad: a single
                #   forward pass on states_seq[t*] under enable_grad() yields
                #   exactly the one loss tensor needed for backward().
                #
                # EAT two-step averaging rule (unchanged):
                #   Trigger if mean(a_{t-1}, a_t) > threshold where
                #   a_t = E_t - 2*E_{t-1} + E_{t-2}.
                #   On trigger: train on step t-1 (first of the pair).
                # ----------------------------------------------------------

                # --- Phase 1: no_grad dry-run to select target_N -----------
                states_seq = []   # states_seq[t] = input state for step t
                errors = []
                prev_accel = None
                target_N = -1     # sentinel: no valid step found yet

                with torch.no_grad():
                    state = x_batch
                    for t in range(max_rollout):
                        states_seq.append(state)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                            pred_dry = model(state, class_labels=labels).sample
                        pred_dry = pred_dry.float()

                        E_t = F.mse_loss(pred_dry, y_seq[:, t]).item()
                        if not math.isfinite(E_t):
                            break

                        errors.append(E_t)
                        target_N = t

                        # EAT check: need at least 3 errors for second-difference.
                        # Normalise by the local window mean to get a dimensionless
                        # curvature fraction — scale-invariant as MSE drops over training.
                        if t >= 2:
                            e0, e1, e2 = errors[-3], errors[-2], errors[-1]
                            window_mean = (e0 + e1 + e2) / 3.0
                            curr_accel = (e2 - 2.0 * e1 + e0) / max(window_mean, 1e-12)
                            if prev_accel is not None:
                                avg_accel = 0.5 * (prev_accel + curr_accel)
                                if avg_accel > current_threshold:
                                    target_N = t - 1  # backprop through first of the two
                                    break
                            prev_accel = curr_accel

                        # Advance rollout state WITH mask (obstacle enforcement only)
                        state = torch.lerp(zero_norm, pred_dry, mask)

                # --- Phase 2: single re-materialised forward with grad -----
                loss_to_backprop = None
                if target_N >= 0:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                        pred_raw = model(states_seq[target_N], class_labels=labels).sample
                    pred_raw = pred_raw.float()
                    loss_to_backprop = F.mse_loss(pred_raw, y_seq[:, target_N])

                if loss_to_backprop is not None:
                    (loss_to_backprop / ACCUM_GRAD).backward()

                if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if target_N >= 0:
                    loss_accum["mse"] += loss_to_backprop.detach() * batch_size
                    loss_accum["N"] += target_N * batch_size
                    # Mean relative acceleration over the observed rollout window.
                    n_e = len(errors)
                    if n_e >= 3:
                        rel_accels = []
                        for i in range(2, n_e):
                            e0, e1, e2 = errors[i - 2], errors[i - 1], errors[i]
                            wm = (e0 + e1 + e2) / 3.0
                            if wm > 1e-12:
                                rel_accels.append((e2 - 2.0 * e1 + e0) / wm)
                        avg_a = sum(rel_accels) / len(rel_accels) if rel_accels else 0.0
                    else:
                        avg_a = 0.0
                    loss_accum["accel"] += avg_a * batch_size
                    sample_count += batch_size

            else:
                # Validation: single-step MSE (no mask in loss)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred_raw = model(x_batch, class_labels=labels).sample
                pred_raw = pred_raw.float()

                y_0 = y_seq[:, 0]
                mse_loss = F.mse_loss(pred_raw, y_0)
                loss_accum["mse"] += mse_loss.detach() * batch_size
                sample_count += batch_size

            # Single Importance reading per iteration is sufficient
            if training:
                loss_accum["importance"] = model.model.latent.gamma.detach() * sample_count

            if step == 0 or (step + 1) % TQDM_UPDATE_EVERY == 0 or (step + 1) == len(loader):
                update_progress(progress_bar, loss_accum, sample_count)

    return {
        "mse":   (loss_accum["mse"] / max(1, sample_count)).detach().item(),
        "N":     loss_accum["N"]     / max(1, sample_count) if training else 0.0,
        "accel": loss_accum["accel"] / max(1, sample_count) if training else 0.0,
        "importance": (loss_accum["importance"] / max(1, sample_count)).detach().item() if training else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
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
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, format="%.3f")
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
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}/")

    print("\nDiscovering simulation folders ...")
    sim_infos = discover_simulations(SIM_ROOT)
    print(f"Found {len(sim_infos)} simulations under {SIM_ROOT}")
    total_frames = sum(sim["n_frames"] for sim in sim_infos)
    print(f"Total frames across all simulations: {total_frames}")

    print("\nEnsuring packed simulation caches ...")
    ensure_all_sim_caches(sim_infos, CACHE_WORKERS, CACHE_STATES_FILENAME, CACHE_MASK_FILENAME)

    train_sim_infos, val_sim_infos = split_simulations(sim_infos, VAL_FRAC)
    print(f"Train simulations: {len(train_sim_infos)}   Val simulations: {len(val_sim_infos)}")

    print("Computing normalization statistics ...")
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

    print("\nBuilding PDE-Transformer (MC-S) ...")
    model = get_model()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total parameters : {n_params:.1f} M")
    print(f"  Trainable params : {n_trainable:.1f} M  (seq_model only)")
    if DEVICE == "cuda":
        print("  Device prefetch: enabled")

    # Load pre-trained base checkpoint BEFORE calibration so the calibration
    # reflects the actual model's error dynamics.
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

    resume_path = RESUME_CHECKPOINT
    if resume_path and os.path.exists(resume_path):
        checkpoint_info = load_checkpoint(model, optimizer, scheduler, resume_path)
        resumed = checkpoint_info["resumed"]
        saved_epoch = checkpoint_info["saved_epoch"]
        best_val = checkpoint_info["best_val"]
        print(f"Loaded checkpoint: {resume_path}")
        if resumed:
            print(
                f"Successfully loaded base checkpoint weights. "
                f"Starting fresh fine-tuning at epoch 1."
            )
        else:
            print("Loaded model weights; optimizer/scheduler state not found.")

    # -----------------------------------------------------------------------
    # Warm-up: force any lazy buffers (e.g. freqs_cis in hat_attn) to
    # initialise NOW in normal (non-inference) mode.
    # If those buffers were first created inside torch.inference_mode() they
    # would be permanently tagged as inference tensors, breaking backward().
    # One cheap no_grad forward pass is sufficient to trigger all lazy inits.
    # -----------------------------------------------------------------------
    print("\nWarm-up pass (lazy buffer initialisation) ...")
    model.eval()
    with torch.no_grad():
        _dummy_in = torch.zeros(1, 3, 256, 256, device=DEVICE)
        _ = model(_dummy_in, class_labels=get_labels(1))
        del _dummy_in, _
    torch.cuda.empty_cache()
    print("  Done.")

    # Calibrate EAT threshold from actual model dynamics on train set
    calibrate_accel_threshold(model, train_loader, zero_norm, get_labels)

    # Post-calibration warm-up: torch.inference_mode() inside calibration marks
    # internal buffers (e.g. freqs_cis) as inference tensors.  A single forward
    # pass with autograd enabled flushes those marks so training backward() works.
    print("\nPost-calibration warm-up (de-tainting inference tensors) ...")
    model.eval()
    model.model.latent.seq_model.train()
    with torch.enable_grad():
        _dummy_in = torch.zeros(1, 3, 256, 256, device=DEVICE, requires_grad=False)
        _dummy_out = model(_dummy_in, class_labels=get_labels(1)).sample
        _dummy_out.sum().backward()
        del _dummy_in, _dummy_out
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    print("  Done.")

    if not SKIP_TRAIN:
        loss_log_path = os.path.join(OUT_DIR, f"loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(loss_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "timestamp epoch lr "
                "train_mse train_N train_accel train_imp "
                "val_mse best\n"
            )

        for epoch in range(start_epoch, EPOCHS + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch:02d}/{EPOCHS}  LR: {current_lr:.2e}  "
                  f"EAT_eps={ACCEL_EPSILON:.3e}  EAT_std={ACCEL_STD:.3e}")

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
                val_loss = {"mse": float("nan")}
            else:
                val_loss = run_epoch(
                    model=model,
                    loader=val_loader,
                    zero_norm=zero_norm,
                    get_labels_fn=get_labels,
                    training=False,
                )

            scheduler.step()
            train_losses["mse"].append(train_loss["mse"])
            val_losses["mse"].append(val_loss["mse"])

            if val_loss["mse"] < best_val:
                best_val = val_loss["mse"]
                best_marker = " ← best mse"
            else:
                best_marker = ""

            save_checkpoint(model, optimizer, scheduler, epoch, best_val, train_losses, val_losses)

            if epoch % 2 == 0:
                epoch_video_path = os.path.join(OUT_DIR, f"pred_vs_gt_epoch_{epoch:03d}.mp4")
                video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
                save_rollout_video(model, video_sim, mean, std, zero_norm, get_labels, epoch_video_path, f"Epoch {epoch:03d}")

            print(
                f"  train: mse={train_loss['mse']:.6f}\n"
                f"         avg rollout N={train_loss['N']:.2f}  max={MAX_ROLLOUT_LEN}  avg_accel={train_loss['accel']:.6f}  imp={train_loss['importance']:.4f}\n"
                f"  val  : mse={val_loss['mse']:.6f}{best_marker}\n"
                f"  cuda mem: {torch.cuda.memory_allocated() / 1e9:.2f}GB / "
                f"max {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            )

            epoch_log_note = "best" if best_marker else ""
            if loss_log_path:
                with open(loss_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{datetime.now().isoformat()} epoch={epoch} lr={current_lr:.2e} "
                        f"train_mse={train_loss['mse']:.6f} train_N={train_loss['N']:.2f} "
                        f"train_accel={train_loss['accel']:.6f} train_imp={train_loss['importance']:.4f} "
                        f"val_mse={val_loss['mse']:.6f} {epoch_log_note}\n"
                    )

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
        print("\nSkipping training (SKIP_TRAIN=True).")

    print("\nGenerating prediction vs ground-truth rollout video ...")

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
    frame_path = save_rollout_video(
        model, video_sim, mean, std, zero_norm, get_labels,
        os.path.join(OUT_DIR, "pred_vs_gt.mp4"), "Final"
    )

    if not SKIP_TRAIN:
        print(f"\n{'=' * 80}")
        print("Fine-tuning complete.")
        print(f"  Best val mse   : {best_val:.6f}")
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
            print(f"{epoch_idx:>6}  {t_m:>11.6f}  {v_m:>11.6f}")
    else:
        print(f"\n{'=' * 80}")
        print("Visualization complete.")
        print(f"  Prediction   : {frame_path}")
        print(f"  Checkpoint   : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(130)