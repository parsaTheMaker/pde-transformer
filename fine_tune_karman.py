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
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import transformers
from pdetransformer.core.sub_network.llm import SequentialModel

# Loss configuration
DIRECTION_LOSS_EPS = 1e-8
DIRECTION_MASK_SMOOTH_PASSES = 0
DIRECTION_MASK_SMOOTH_KERNEL = 5
LOSS_WEIGHT_MSE = 1.0
LOSS_WEIGHT_GRAD = 2.0
LOSS_WEIGHT_DIRECTION = 0.02

# LLM Sub-Network Configuration
SEQ_HIDDEN_SIZE = 144
SEQ_NUM_HEADS = 8
SEQ_N_LAYERS = 4
# Options for SEQ_ATTN_METHOD: 
# - "hyper": uses HyperAttention (requires Triton, Linux only)
# - "naive": uses vanilla PyTorch scaled_dot_product_attention (faster, cross-platform)
SEQ_ATTN_METHOD = "naive"

# Error-Acceleration Truncation Configuration
MAX_ROLLOUT_LEN = 10
ACCEL_EPSILON = 0.000055
# Use "mse" for much faster unrolling, or "combo" for full spatial gradient/direction evaluations
TRUNCATION_METRIC = "combo"


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
SIM_ROOT = "/home/vatani/data_vortex/256_inc"  # e.g. r"D:\data\256_inc" or "/data/256_inc"
OUT_DIR = os.path.join("runs", "karman_finetuned")
EPOCHS = 40
BATCH_SIZE = 20
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
RESUME_CHECKPOINT = "./runs/karman/last.ckpt"  # e.g. r"D:\runs\karman\last.ckpt" or "/home/user/last.ckpt"
MODEL_TYPE = "PDE-S"  # Smallest PDETransformer variant in this repo
USE_AMP = DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
# This model is attention-heavy and does many explicit permutes/window reshapes,
# so channels_last is not a reliable speedup here.
USE_CHANNELS_LAST = False
NUM_WORKERS = max(0, cpu_count() - 4)
PIN_MEMORY = DEVICE == "cuda"
CACHE_STATES_FILENAME = "states.float32.npy"
CACHE_MASK_FILENAME = "obstacle_mask.float32.npy"
CACHE_WORKERS = max(1, cpu_count() - 4)
PREFETCH_FACTOR = 2
TQDM_UPDATE_EVERY = 20

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
        y_seq_np = packed_slice_to_numpy(states[frame_idx + 1 : end_idx])
        y_seq_np = np.asarray(y_seq_np, dtype=np.float32)
        np.subtract(y_seq_np, self.mean_np, out=y_seq_np)
        np.multiply(y_seq_np, self.inv_std_np, out=y_seq_np)
        y_seq = torch.from_numpy(y_seq_np)

        return x.float(), y_seq.float(), self._get_mask(sim_idx)


class GradMagAndDirectionLoss(nn.Module):
    def __init__(self, eps=1e-8, dir_tau=1e-4, use_smooth_dir_weight=True, mask_smooth_passes=3, mask_smooth_kernel=5):
        super().__init__()
        self.eps = float(eps)
        self.dir_tau = float(dir_tau)
        self.use_smooth_dir_weight = bool(use_smooth_dir_weight)
        
        self.mask_smooth_passes = int(mask_smooth_passes)
        self.mask_smooth_kernel = int(mask_smooth_kernel)

    def _prepare_mask(self, valid_mask, x):
        if valid_mask is None:
            return None
        
        vm = valid_mask.to(device=x.device, dtype=x.dtype)
        if vm.ndim == x.ndim - 1:
            vm = vm.unsqueeze(1)
        if vm.ndim != x.ndim:
            raise ValueError(f"Mask ndim {vm.ndim} incompatible with input ndim {x.ndim}")
        if vm.shape[0] != x.shape[0] or vm.shape[2:] != x.shape[2:]:
            raise ValueError(f"Mask shape {vm.shape} incompatible with input shape {x.shape}")
            
        if self.mask_smooth_passes > 0:
            kernel = self.mask_smooth_kernel
            padding = kernel // 2
            if x.ndim == 4:
                for _ in range(self.mask_smooth_passes):
                    vm = F.avg_pool2d(vm, kernel_size=kernel, stride=1, padding=padding)
            elif x.ndim == 5:
                for _ in range(self.mask_smooth_passes):
                    vm = F.avg_pool3d(vm, kernel_size=kernel, stride=1, padding=padding)
        
        vm = vm.clamp(0.0, 1.0)
        if vm.shape[1] == 1 and x.shape[1] != 1:
            vm = vm.expand(-1, x.shape[1], *x.shape[2:])
        elif vm.shape[1] != x.shape[1]:
            raise ValueError(f"Mask channels {vm.shape[1]} must be 1 or match input channels {x.shape[1]}")
        return vm

    def get_components(self, pred, target, valid_mask=None, spacing=None):
        pred_grads = self._spatial_grads(pred, spacing=spacing)
        target_grads = self._spatial_grads(target, spacing=spacing)

        pred_g = torch.stack(pred_grads, dim=2)
        target_g = torch.stack(target_grads, dim=2)

        m = self._prepare_mask(valid_mask, pred)

        # Gradient Magnitude / Sobolev term
        grad_diff_sq = (pred_g - target_g).square().sum(dim=2)
        if m is not None:
            grad_loss = (grad_diff_sq * m).sum() / (m.sum() + self.eps)
        else:
            grad_loss = grad_diff_sq.mean()

        # Gradient Direction term
        dot = (pred_g * target_g).sum(dim=2)
        pred_norm = torch.sqrt(pred_g.square().sum(dim=2) + self.eps)
        target_norm = torch.sqrt(target_g.square().sum(dim=2) + self.eps)

        cos = dot / (pred_norm * target_norm + self.eps)
        dir_loss_pointwise = 1.0 - cos

        if self.use_smooth_dir_weight:
            w_dir = target_norm / (target_norm + self.dir_tau)
        else:
            w_dir = (target_norm > self.dir_tau).to(dtype=pred.dtype)

        if m is not None:
            w = w_dir * m
            dir_loss = (dir_loss_pointwise * w).sum() / (w.sum() + self.eps)
        else:
            dir_loss = (dir_loss_pointwise * w_dir).sum() / (w_dir.sum() + self.eps)

        return grad_loss, dir_loss

    def grad_only(self, pred, target, valid_mask=None, spacing=None):
        grad_loss, _ = self.get_components(pred, target, valid_mask, spacing)
        return grad_loss

    def direction_only(self, pred, target, valid_mask=None, spacing=None):
        _, dir_loss = self.get_components(pred, target, valid_mask, spacing)
        return dir_loss

    def _spatial_grads(self, x: torch.Tensor, spacing=None):
        if x.ndim == 4:
            return self._grads_2d(x, spacing)
        elif x.ndim == 5:
            return self._grads_3d(x, spacing)
        raise ValueError("Expected [B, C, H, W] or [B, C, D, H, W]")

    def _grads_2d(self, x: torch.Tensor, spacing=None):
        if spacing is None:
            dy, dx = 1.0, 1.0
        else:
            dy, dx = float(spacing[0]), float(spacing[1])

        gy = torch.empty_like(x)
        gx = torch.empty_like(x)

        gy[:, :, 1:-1, :] = (x[:, :, 2:, :] - x[:, :, :-2, :]) / (2.0 * dy)
        gy[:, :, 0, :] = (x[:, :, 1, :] - x[:, :, 0, :]) / dy
        gy[:, :, -1, :] = (x[:, :, -1, :] - x[:, :, -2, :]) / dy

        gx[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / (2.0 * dx)
        gx[:, :, :, 0] = (x[:, :, :, 1] - x[:, :, :, 0]) / dx
        gx[:, :, :, -1] = (x[:, :, :, -1] - x[:, :, :, -2]) / dx

        return gy, gx

    def _grads_3d(self, x: torch.Tensor, spacing=None):
        if spacing is None:
            dz, dy, dx = 1.0, 1.0, 1.0
        else:
            dz, dy, dx = float(spacing[0]), float(spacing[1]), float(spacing[2])

        gz = torch.empty_like(x)
        gy = torch.empty_like(x)
        gx = torch.empty_like(x)

        gz[:, :, 1:-1, :, :] = (x[:, :, 2:, :, :] - x[:, :, :-2, :, :]) / (2.0 * dz)
        gz[:, :, 0, :, :] = (x[:, :, 1, :, :] - x[:, :, 0, :, :]) / dz
        gz[:, :, -1, :, :] = (x[:, :, -1, :, :] - x[:, :, -2, :, :]) / dz

        gy[:, :, :, 1:-1, :] = (x[:, :, :, 2:, :] - x[:, :, :, :-2, :]) / (2.0 * dy)
        gy[:, :, :, 0, :] = (x[:, :, :, 1, :] - x[:, :, :, 0, :]) / dy
        gy[:, :, :, -1, :] = (x[:, :, :, -1, :] - x[:, :, :, -2, :]) / dy

        gx[:, :, :, :, 1:-1] = (x[:, :, :, :, 2:] - x[:, :, :, :, :-2]) / (2.0 * dx)
        gx[:, :, :, :, 0] = (x[:, :, :, :, 1] - x[:, :, :, :, 0]) / dx
        gx[:, :, :, :, -1] = (x[:, :, :, :, -1] - x[:, :, :, :, -2]) / dx

        return gz, gy, gx


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



class LatentWrapper(nn.Module):
    def __init__(self, orig_latent, seq_model):
        super().__init__()
        self.orig_latent = orig_latent
        self.seq_model = seq_model
    
    def forward(self, x, c):
        x = self.orig_latent(x, c)
        original_dtype = x.dtype
        x_5d = x.unsqueeze(-1) # [N, C, H, W, 1]
        out_5d = self.seq_model(x_5d)
        out_4d = out_5d.squeeze(-1)
        x = x + out_4d
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
        use_conv_proj=False
    ).to(DEVICE)
    
    if USE_CHANNELS_LAST:
        seq_model = seq_model.to(memory_format=torch.channels_last)
    
    base_model.model.latent = LatentWrapper(base_model.model.latent, seq_model)
    
    return base_model

def create_optimizer(model):
    # Only optimize seq_model parameters
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
    avg_loss_dict = {}
    for key, value in loss_dict.items():
        avg = value / denom
        if isinstance(avg, torch.Tensor):
            avg = avg.detach().item()
        avg_loss_dict[key] = avg
    progress_bar.set_postfix(
        mse=f"{avg_loss_dict['mse']:.6f}",
        grad=f"{avg_loss_dict['grad']:.6f}",
        dir=f"{avg_loss_dict['direction']:.6f}",
        combo=f"{avg_loss_dict['combo']:.6f}",
        N=f"{avg_loss_dict.get('N', 0):.2f}",
        accel=f"{avg_loss_dict.get('accel', 0):.6f}"
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


def normalize_loss_history(losses):
    normalized = {"mse": [], "grad": [], "direction": [], "combo": []}
    if not isinstance(losses, dict):
        return normalized

    mse_hist = list(losses.get("mse", []))
    grad_hist = list(losses.get("grad", []))
    direction_hist = list(losses.get("direction", []))
    combo_hist = list(losses.get("combo", []))

    length = max(len(mse_hist), len(grad_hist), len(direction_hist), len(combo_hist))
    for key, hist in (("mse", mse_hist), ("grad", grad_hist), ("direction", direction_hist), ("combo", combo_hist)):
        if len(hist) < length:
            hist.extend([float("nan")] * (length - len(hist)))
        normalized[key] = hist

    for idx in range(length):
        if math.isnan(normalized["combo"][idx]):
            normalized["combo"][idx] = (
                LOSS_WEIGHT_MSE * normalized["mse"][idx]
                + LOSS_WEIGHT_GRAD * normalized["grad"][idx]
                + LOSS_WEIGHT_DIRECTION * normalized["direction"][idx]
            )

    return normalized


def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses):
    checkpoint = build_checkpoint(model, optimizer, scheduler, epoch, best_metric, train_losses, val_losses)
    torch.save(checkpoint, os.path.join(OUT_DIR, "last.ckpt"))


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Adapt state_dict for LatentWrapper
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.latent.") and not k.startswith("model.latent.orig_latent.") and not k.startswith("model.latent.seq_model."):
            new_k = k.replace("model.latent.", "model.latent.orig_latent.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # We purposely do NOT load optimizer and scheduler states because 
        # we are fine-tuning a new sub-network and keeping the old model frozen.
        return {
            "resumed": True,
            "saved_epoch": int(checkpoint.get("epoch", 0)),
            "best_val": float(checkpoint.get("best_val", math.inf)),
            "train_losses": {"mse": [], "grad": [], "direction": [], "combo": []},
            "val_losses": {"mse": [], "grad": [], "direction": [], "combo": []},
        }

    return {
        "resumed": False,
        "saved_epoch": None,
        "best_val": math.inf,
        "train_losses": {"mse": [], "grad": [], "direction": [], "combo": []},
        "val_losses": {"mse": [], "grad": [], "direction": [], "combo": []},
    }


def fast_get_gradient_vector(network):
    """Optimized method to extract gradients without creating O(N) memory fragments."""
    with torch.no_grad():
        grads = [par.grad.data.view(-1) for par in network.parameters() if par.grad is not None]
        return torch.cat(grads) if grads else None


def sanitize_gradient_vector(grad_vec):
    """Return a finite gradient vector or None if no usable gradient exists."""
    if grad_vec is None:
        return None
    if torch.isfinite(grad_vec).all():
        return grad_vec
    return torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)


def run_epoch(model, loader, zero_norm, get_labels_fn, loss_fn, training, optimizer=None, operator=None, global_step=0, step_limit=None):
    if training:
        # Base model frozen -> strictly eval mode. Layers natively drop performance if tracked natively in train mode.
        model.eval() 
        model.model.latent.seq_model.train() 
        optimizer.zero_grad(set_to_none=True)
        desc = "tr"
    else:
        model.eval()
        desc = "va"

    loss_accum = {
        "mse": torch.zeros((), device=DEVICE),
        "grad": torch.zeros((), device=DEVICE),
        "direction": torch.zeros((), device=DEVICE),
        "combo": torch.zeros((), device=DEVICE),
        "N": torch.zeros((), device=DEVICE),
        "accel": torch.zeros((), device=DEVICE),
    }
    sample_count = 0

    progress_bar = make_progress(loader, desc)
    for step, (x_batch, y_seq, mask) in enumerate(progress_bar):
        x_batch, y_seq, mask = move_batch_to_device(x_batch, y_seq, mask)
        labels = get_labels_fn(x_batch.shape[0])
        batch_size = x_batch.shape[0]

        max_rollout = min(MAX_ROLLOUT_LEN, y_seq.shape[1])
        current_accel_epsilon = max(0.0, float(np.random.normal(loc=ACCEL_EPSILON, scale=0.20 * ACCEL_EPSILON)))
        
        target_N = None
        state_N_minus_1 = None
        state = x_batch.clone() 
        errors = []

        if training:
            # Single-pass unrolling with gradients evaluated locally for optimal truncation capture
            target_N = max_rollout - 1
            loss_to_backprop = None
            mse_loss_out = dir_loss_out = grad_loss_out = total_loss_out = None
            max_a_t = 0.0
            
            for t in range(max_rollout):
                state_in = state.detach()
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                    pred = model(state_in, class_labels=labels).sample
                pred = pred.float()
                pred = torch.lerp(zero_norm, pred, mask)
                
                y_t = y_seq[:, t]
                mse_loss = F.mse_loss(pred, y_t)
                
                if TRUNCATION_METRIC == "mse":
                    total_loss = mse_loss
                    grad_loss, dir_loss = torch.tensor(0.0, device=DEVICE), torch.tensor(0.0, device=DEVICE)
                else:
                    grad_loss, dir_loss = loss_fn.get_components(pred, y_t, valid_mask=mask)
                    total_loss = mse_loss * LOSS_WEIGHT_MSE + grad_loss * LOSS_WEIGHT_GRAD + dir_loss * LOSS_WEIGHT_DIRECTION
                
                E_t_val = total_loss.item()
                if not math.isfinite(E_t_val):
                    loss_to_backprop = None
                    break
                errors.append(E_t_val)
                
                # Assign tracker values
                mse_loss_out, grad_loss_out, dir_loss_out, total_loss_out = mse_loss, grad_loss, dir_loss, total_loss
                loss_to_backprop = total_loss
                target_N = t
                
                if t >= 2:
                    a_t = E_t_val - 2 * errors[-2] + errors[-3]
                    max_a_t = max(max_a_t, a_t)
                    if a_t > current_accel_epsilon:
                        # Reached acceleration limit; trigger drift backprop & halt further execution
                        break
                        
                state = pred 
                
            if loss_to_backprop is not None:
                loss_to_backprop = loss_to_backprop / ACCUM_GRAD
                loss_to_backprop.backward()
            
            if (step + 1) % ACCUM_GRAD == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if loss_to_backprop is not None:
                loss_accum["mse"] += mse_loss_out.detach() * batch_size
                loss_accum["grad"] += grad_loss_out.detach() * batch_size
                loss_accum["direction"] += dir_loss_out.detach() * batch_size
                loss_accum["combo"] += total_loss_out.detach() * batch_size
                loss_accum["N"] += target_N * batch_size
                loss_accum["accel"] += max_a_t * batch_size
                sample_count += batch_size
            
        else: # eval
            with torch.inference_mode():
                pred = model(x_batch, class_labels=labels).sample.float()
                pred = torch.lerp(zero_norm, pred, mask)
                y_0 = y_seq[:, 0]
                mse_loss = F.mse_loss(pred, y_0)
                grad_loss, dir_loss = loss_fn.get_components(pred, y_0, valid_mask=mask)
                total_loss = mse_loss * LOSS_WEIGHT_MSE + grad_loss * LOSS_WEIGHT_GRAD + dir_loss * LOSS_WEIGHT_DIRECTION
                
                loss_accum["mse"] += mse_loss.detach() * batch_size
                loss_accum["grad"] += grad_loss.detach() * batch_size
                loss_accum["direction"] += dir_loss.detach() * batch_size
                loss_accum["combo"] += total_loss.detach() * batch_size
                sample_count += batch_size

        if step == 0 or (step + 1) % TQDM_UPDATE_EVERY == 0 or (step + 1) == len(loader):
            update_progress(progress_bar, loss_accum, sample_count)
            
        if training and step_limit is not None and step >= step_limit - 1:
            break

    return {
        "mse": (loss_accum["mse"] / max(1, sample_count)).detach().item(),
        "grad": (loss_accum["grad"] / max(1, sample_count)).detach().item(),
        "direction": (loss_accum["direction"] / max(1, sample_count)).detach().item(),
        "combo": (loss_accum["combo"] / max(1, sample_count)).detach().item(),
        "N": (loss_accum["N"] / max(1, sample_count)).detach().item() if training else 0,
        "accel": (loss_accum["accel"] / max(1, sample_count)).detach().item() if training else 0,
    }

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
            pres(gt_norm_seq[idx]), pres(pred_frames[idx]), err_pres
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
            0.5,
            0.94,
            f"{title_tag}   |   t = {start_idx + idx:05d}   |   Frame {idx:03d}/{len(pred_frames) - 1}",
            ha="center",
            va="top",
            color="white",
            fontsize=9,
            fontfamily="monospace",
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

    # Build loss function
    loss_fn = GradMagAndDirectionLoss(
        eps=DIRECTION_LOSS_EPS,
        mask_smooth_passes=DIRECTION_MASK_SMOOTH_PASSES,
        mask_smooth_kernel=DIRECTION_MASK_SMOOTH_KERNEL,
    ).to(DEVICE)
    print(f"\nLoss: GradMagAndDirectionLoss")
    print(f"  Direction mask smoothing: passes={loss_fn.mask_smooth_passes}, kernel={loss_fn.mask_smooth_kernel}")
    print(
        "  Combo weights: "
        f"mse={LOSS_WEIGHT_MSE:g}, grad={LOSS_WEIGHT_GRAD:g}, dir={LOSS_WEIGHT_DIRECTION:g}"
    )

    task_label = torch.tensor([1000], dtype=torch.long, device=DEVICE)

    def get_labels(batch_size):
        return task_label.expand(batch_size)

    from conflictfree.momentum_operator import PseudoMomentumOperator
    operator = PseudoMomentumOperator(num_vectors=3)
    global_step = 0

    train_losses = {"mse": [], "grad": [], "direction": [], "combo": []}
    val_losses = {"mse": [], "grad": [], "direction": [], "combo": []}
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
                f"Starting completely fresh fine-tuning at epoch 1."
            )
        else:
            print("Successfully loaded model weights from checkpoint; optimizer/scheduler state not found.")

    if not SKIP_TRAIN:
        loss_log_path = os.path.join(OUT_DIR, f"loss_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(loss_log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                "timestamp epoch lr "
                "train_mse train_grad train_direction train_combo train_N train_accel "
                "val_mse val_grad val_direction val_combo best\n"
            )

        for epoch in range(start_epoch, EPOCHS + 1):
            # Print current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch:02d}/{EPOCHS}  LR: {current_lr:.2e}")

            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                zero_norm=zero_norm,
                get_labels_fn=get_labels,
                loss_fn=loss_fn,
                training=True,
                optimizer=optimizer,
                operator=operator,
                global_step=global_step,
            )
            global_step += len(train_loader)

            if val_loader is None:
                val_loss = {"mse": float("nan"), "grad": float("nan"), "direction": float("nan"), "combo": float("nan")}
            else:
                val_loss = run_epoch(
                    model=model,
                    loader=val_loader,
                    zero_norm=zero_norm,
                    get_labels_fn=get_labels,
                    loss_fn=loss_fn,
                    training=False,
                )

            scheduler.step()
            train_losses["mse"].append(train_loss["mse"])
            train_losses["grad"].append(train_loss["grad"])
            train_losses["direction"].append(train_loss["direction"])
            train_losses["combo"].append(train_loss["combo"])
            
            val_losses["mse"].append(val_loss["mse"])
            val_losses["grad"].append(val_loss["grad"])
            val_losses["direction"].append(val_loss["direction"])
            val_losses["combo"].append(val_loss["combo"])

            if val_loss["combo"] < best_val:
                best_val = val_loss["combo"]
                best_marker = " ← best combo"
            else:
                best_marker = ""

            save_checkpoint(model, optimizer, scheduler, epoch, best_val, train_losses, val_losses)

            if epoch % 2 == 0:
                epoch_video_path = os.path.join(OUT_DIR, f"pred_vs_gt_epoch_{epoch:03d}.mp4")
                video_sim = val_sim_infos[0] if val_sim_infos else train_sim_infos[0]
                save_rollout_video(model, video_sim, mean, std, zero_norm, get_labels, epoch_video_path, f"Epoch {epoch:03d}")

            print(
                f"  train: mse={train_loss['mse']:.6f} grad={train_loss['grad']:.6f} dir={train_loss['direction']:.6f} combo={train_loss['combo']:.6f} \n"
                f"         avg rollout N={train_loss['N']:.2f} max_rollout={MAX_ROLLOUT_LEN} avg_accel={train_loss['accel']:.6f}\n"
                f"  val  : mse={val_loss['mse']:.6f} grad={val_loss['grad']:.6f} dir={val_loss['direction']:.6f} combo={val_loss['combo']:.6f}{best_marker}\n"
                f"  cuda mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB / max: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            )

            epoch_log_note = "best" if best_marker else ""
            if loss_log_path:
                log_line = (
                    f"{datetime.now().isoformat()} epoch={epoch} lr={current_lr:.2e} "
                    f"train_mse={train_loss['mse']:.6f} train_grad={train_loss['grad']:.6f} train_dir={train_loss['direction']:.6f} train_combo={train_loss['combo']:.6f} train_N={train_loss['N']:.2f} train_accel={train_loss['accel']:.6f} "
                    f"val_mse={val_loss['mse']:.6f} val_grad={val_loss['grad']:.6f} val_dir={val_loss['direction']:.6f} val_combo={val_loss['combo']:.6f} {epoch_log_note}\n"
                )
                with open(loss_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(log_line)

        print("\nSaving loss curves …")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#111")
        axes = axes.ravel()
        
        # Combo loss plot
        ax = axes[0]
        ax.set_facecolor("#111")
        ax.plot(range(1, len(train_losses['combo']) + 1), train_losses['combo'], color="#aa88ff", linewidth=2, label="Train Combo")
        ax.plot(range(1, len(val_losses['combo']) + 1), val_losses['combo'], color="#ff6b35", linewidth=2, label="Val Combo")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Loss", color="white")
        ax.set_title("Combo Weighted Loss", color="white", fontsize=11)
        ax.legend(framealpha=0.3)
        ax.tick_params(colors="white")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
            
        # MSE loss plot
        ax = axes[1]
        ax.set_facecolor("#111")
        ax.plot(range(1, len(train_losses['mse']) + 1), train_losses['mse'], color="#00c8ff", linewidth=2, label="Train MSE")
        ax.plot(range(1, len(val_losses['mse']) + 1), val_losses['mse'], color="#ffaa00", linewidth=2, label="Val MSE")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("MSE Loss", color="white")
        ax.set_title("MSE Loss", color="white", fontsize=11)
        ax.legend(framealpha=0.3)
        ax.tick_params(colors="white")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        # Grad loss plot
        ax = axes[2]
        ax.set_facecolor("#111")
        ax.plot(range(1, len(train_losses['grad']) + 1), train_losses['grad'], color="#ff33cc", linewidth=2, label="Train Grad")
        ax.plot(range(1, len(val_losses['grad']) + 1), val_losses['grad'], color="#ffaa00", linewidth=2, label="Val Grad")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Grad Magnitude Loss", color="white")
        ax.set_title("Grad Magnitude Loss", color="white", fontsize=11)
        ax.legend(framealpha=0.3)
        ax.tick_params(colors="white")
        ax.set_yscale("log")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

        # Direction loss plot
        ax = axes[3]
        ax.set_facecolor("#111")
        ax.plot(range(1, len(train_losses['direction']) + 1), train_losses['direction'], color="#00ff88", linewidth=2, label="Train Direction")
        ax.plot(range(1, len(val_losses['direction']) + 1), val_losses['direction'], color="#ffafcc", linewidth=2, label="Val Direction")
        ax.set_xlabel("Epoch", color="white")
        ax.set_ylabel("Direction Loss", color="white")
        ax.set_title("Direction Loss", color="white", fontsize=11)
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
        print(f"\n{'=' * 80}")
        print("Training complete.")
        print(f"  Best val combo : {best_val:.6f}")
        if loss_log_path:
            print(f"  Loss log       : {loss_log_path}")
        print(f"  Loss curves    : {OUT_DIR}/loss_curves.png")
        print(f"  Prediction     : {frame_path}")
        print(f"  Checkpoint     : {OUT_DIR}/last.ckpt")
        print(f"{'=' * 80}\n")
        header = (
            f"{'Epoch':>6}  {'Train Combo':>13}  {'Val Combo':>13}  "
            f"{'Train MSE':>11}  {'Val MSE':>11}"
        )
        print(header)
        print("-" * len(header))
        for epoch_idx, (t_c, v_c, t_m, v_m) in enumerate(
            zip(train_losses['combo'], val_losses['combo'], train_losses['mse'], val_losses['mse']), 1
        ):
            print(
                f"{epoch_idx:>6}  {t_c:>13.6f}  {v_c:>13.6f}  "
                f"{t_m:>11.6f}  {v_m:>11.6f}"
            )
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