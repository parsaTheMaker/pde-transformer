"""
EAT.py
======
Reusable implementation of an epoch-local rollout curriculum for
autoregressive neural simulators.

EAT means Error-Accelaration Aware Training in this module: the model is
trained autoregressively with full BPTT over a curriculum frontier N, and the
frontier is promoted when the positive error velocity at that frontier improves
enough.

The module intentionally does not know about PDE-Transformer, LoRA, Karman
datasets, cache files, videos, or plotting. It only needs tensors and a forward
function. This makes it usable with other models and training scripts.

Expected batch contract
-----------------------
    x:     [B, C, H, W]          normalized input state at time t
    y_seq: [B, T, C, H, W]       normalized targets at t+1 ... t+T
    mask:  [B, 1, H, W] or [B,H,W] optional rollout-valid mask

Loss contract
-------------
    mse(pred_raw, y_t)

The mask is never applied to the loss. The mask is applied only when advancing
the autoregressive state:

    next_state = lerp(zero_state, pred_raw, mask)

This is the rollout-update rule used by Error-Accelaration Aware Training.
"""

from __future__ import annotations

from contextlib import nullcontext, suppress
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


ForwardFn = Callable[..., Any]
LabelsFn = Callable[[int, torch.device], Optional[torch.Tensor]]


@dataclass
class EATConfig:
    """Hyperparameters for epoch-local Error-Accelaration Aware Training.

    Args:
        max_rollout_len: Maximum number of one-step targets available per
            sample. If y_seq is shorter, the batch uses the shorter length.
        initial_frontier_n: Epoch-local starting frontier. The recommended
            default is N=1, meaning loss is backpropagated through steps 0 and 1.
        initial_improvement_frac: Required fractional improvement used for the
            first epoch if no persisted value is loaded.
        min_improvement_frac: Lower bound for the next-epoch improvement
            requirement when an epoch cannot reach the full rollout.
        min_stage_batches: Number of frontier-reaching batches required before
            promotion checks can use the prior-stage baseline.
        recent_promo_window: Number of latest frontier-reaching batches compared
            against the prior-stage baseline.
        grad_accum_steps: Gradient accumulation interval.
        grad_clip_norm: Optional global norm clipping before optimizer.step().
        use_amp: Use torch.autocast during forward passes.
        amp_dtype: Autocast dtype, typically torch.bfloat16 on modern CUDA GPUs.
        use_activation_checkpointing: Recompute model activations during
            training backward to reduce VRAM usage.
        preserve_rng_state: Passed to torch.utils.checkpoint.
        apply_mask_to_rollout: Keep this True for obstacle/domain enforcement.
            Set False only for systems where the raw prediction should fully
            replace the state.
        reduce_ddp: If True, scalar velocity and epoch totals are all-reduced
            with torch.distributed when initialized.
        collect_batch_frontier_history: Store per-batch frontier N values for
            plotting curriculum progress.
    """

    max_rollout_len: int = 8
    initial_frontier_n: int = 1
    initial_improvement_frac: float = 0.30
    min_improvement_frac: float = 0.01
    min_stage_batches: int = 30
    recent_promo_window: int = 10
    grad_accum_steps: int = 1
    grad_clip_norm: Optional[float] = 1.0
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.bfloat16
    use_activation_checkpointing: bool = True
    preserve_rng_state: bool = False
    apply_mask_to_rollout: bool = True
    reduce_ddp: bool = True
    collect_batch_frontier_history: bool = True

    def validate(self) -> None:
        if self.max_rollout_len < 1:
            raise ValueError("max_rollout_len must be >= 1")
        if self.initial_frontier_n < 0:
            raise ValueError("initial_frontier_n must be >= 0")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.min_stage_batches < 1:
            raise ValueError("min_stage_batches must be >= 1")
        if self.recent_promo_window < 1:
            raise ValueError("recent_promo_window must be >= 1")
        if not 0.0 <= self.initial_improvement_frac < 1.0:
            raise ValueError("initial_improvement_frac must be in [0, 1)")
        if not 0.0 <= self.min_improvement_frac < 1.0:
            raise ValueError("min_improvement_frac must be in [0, 1)")


@dataclass
class EATState:
    """Checkpointable state for the next epoch.

    EAT resets the frontier to initial_frontier_n at the start of every epoch.
    The adaptive part that persists across epochs is only the promotion
    improvement requirement.
    """

    promotion_improvement_frac_next: float
    epoch: int = 0

    @classmethod
    def from_config(cls, config: EATConfig) -> "EATState":
        return cls(promotion_improvement_frac_next=float(config.initial_improvement_frac))

    def state_dict(self) -> Dict[str, float]:
        return {
            "promotion_improvement_frac_next": float(self.promotion_improvement_frac_next),
            "epoch": int(self.epoch),
        }

    def load_state_dict(self, state: MutableMapping[str, Any]) -> None:
        self.promotion_improvement_frac_next = float(
            state.get("promotion_improvement_frac_next", self.promotion_improvement_frac_next)
        )
        self.epoch = int(state.get("epoch", self.epoch))


@dataclass
class EATStage:
    """Mutable per-epoch curriculum stage state."""

    current_frontier_n: int
    stage_batch_count: int = 0
    stage_frontier_vels: List[float] = field(default_factory=list)
    stage_vel_sum: float = 0.0
    active_baseline_vel: Optional[float] = None
    last_valid_baseline_vel: Optional[float] = None
    promotion_count: int = 0

    @property
    def running_stage_vel(self) -> float:
        if self.stage_batch_count == 0:
            return float("nan")
        return self.stage_vel_sum / float(self.stage_batch_count)

    def observe_frontier_velocity(self, velocity: float, config: EATConfig, improvement_frac: float) -> bool:
        """Record a frontier velocity and promote N if the stage improved.

        Promotion rule:
            mean(last recent_promo_window velocities)
                <= mean(prior velocities) * (1 - improvement_frac)

        The prior baseline is only valid after at least min_stage_batches
        baseline batches exist. On promotion, the stage history resets.
        """

        self.stage_frontier_vels.append(float(velocity))
        self.stage_batch_count += 1
        self.stage_vel_sum += float(velocity)
        self.active_baseline_vel = None

        recent_mean_vel = None
        enough_points = self.stage_batch_count >= config.min_stage_batches + config.recent_promo_window
        if enough_points:
            baseline_values = self.stage_frontier_vels[: -config.recent_promo_window]
            recent_values = self.stage_frontier_vels[-config.recent_promo_window :]
            if len(baseline_values) >= config.min_stage_batches:
                self.active_baseline_vel = float(np.mean(baseline_values, dtype=np.float64))
                self.last_valid_baseline_vel = self.active_baseline_vel
                recent_mean_vel = float(np.mean(recent_values, dtype=np.float64))

        max_frontier_n = config.max_rollout_len - 1
        should_promote = (
            self.current_frontier_n < max_frontier_n
            and self.active_baseline_vel is not None
            and recent_mean_vel is not None
            and recent_mean_vel <= self.active_baseline_vel * (1.0 - float(improvement_frac))
        )
        if not should_promote:
            return False

        self.current_frontier_n += 1
        self.stage_batch_count = 0
        self.stage_frontier_vels = []
        self.stage_vel_sum = 0.0
        self.active_baseline_vel = None
        self.promotion_count += 1
        return True

    def reported_baseline_vel(self) -> Optional[float]:
        if self.active_baseline_vel is not None:
            return self.active_baseline_vel
        return self.last_valid_baseline_vel


@dataclass
class EATBatchResult:
    loss: torch.Tensor
    loss_value: float
    effective_frontier_n: int
    frontier_velocity: Optional[float]
    batch_size: int


@dataclass
class EATEpochResult:
    mse: float
    avg_target_n: float
    train_frontier_vel: float
    frontier_baseline_vel: Optional[float]
    stage_batch_count: int
    promotion_count: int
    promotion_improvement_frac: float
    promotion_improvement_frac_next: float
    train_frontier_n: int
    max_frontier_n: int
    running_stage_frontier_vel: float
    frontier_vel_count: int
    batch_progress: Optional[List[float]] = None
    batch_n_values: Optional[List[float]] = None

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EATEvalResult:
    mse: float
    all_mses: Optional[np.ndarray]
    max_frontier_n: int

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


def default_forward(model: torch.nn.Module, state: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward helper compatible with PDE-Transformer and plain PyTorch models."""

    if labels is None:
        out = model(state)
    else:
        try:
            out = model(state, class_labels=labels)
        except TypeError:
            out = model(state, labels)
    return out.sample if hasattr(out, "sample") else out


def default_labels(batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
    """Default label provider for unconditional models."""

    del batch_size, device
    return None


def move_batch_to_device(
    batch: Sequence[torch.Tensor],
    device: torch.device,
    channels_last: bool = False,
    non_blocking: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Move a `(x, y_seq, mask)` or `(x, y_seq)` batch to the target device."""

    if len(batch) == 2:
        x, y_seq = batch
        mask = None
    elif len(batch) == 3:
        x, y_seq, mask = batch
    else:
        raise ValueError("EAT batches must be (x, y_seq) or (x, y_seq, mask)")

    x = x.to(device, non_blocking=non_blocking)
    y_seq = y_seq.to(device, non_blocking=non_blocking)
    if mask is not None:
        mask = mask.to(device, non_blocking=non_blocking)

    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
        y_seq = y_seq.contiguous(memory_format=torch.channels_last)
        if mask is not None:
            mask = mask.contiguous(memory_format=torch.channels_last)

    return x, y_seq, mask


def normalize_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert `[B,H,W]` masks to `[B,1,H,W]` for broadcasting."""

    if mask is not None and mask.ndim == 3:
        return mask.unsqueeze(1)
    return mask


def advance_rollout_state(
    pred_raw: torch.Tensor,
    mask: Optional[torch.Tensor],
    zero_state: Optional[torch.Tensor],
    apply_mask: bool = True,
) -> torch.Tensor:
    """Advance the autoregressive state using the reference masking rule."""

    if not apply_mask or mask is None:
        return pred_raw
    if zero_state is None:
        raise ValueError("zero_state is required when apply_mask=True and mask is provided")
    return torch.lerp(zero_state, pred_raw, mask)


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE reduced over non-batch dimensions, preserving one value per sample."""

    dims = tuple(range(1, pred.ndim))
    return F.mse_loss(pred, target, reduction="none").mean(dim=dims)


def positive_velocity_sum_count(
    curr_err_per_sample: torch.Tensor,
    prev_err_per_sample: torch.Tensor,
) -> Tuple[float, int]:
    """Sum and count only positive error increments E_t - E_{t-1}."""

    vel = curr_err_per_sample - prev_err_per_sample
    pos_vel = vel[vel > 0]
    return float(pos_vel.sum().item()), int(pos_vel.numel())


def distributed_is_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def global_positive_velocity(
    curr_err_per_sample: torch.Tensor,
    prev_err_per_sample: torch.Tensor,
    reduce_ddp: bool = True,
) -> Tuple[float, float, int]:
    """Mean positive velocity, optionally all-reduced across DDP ranks."""

    local_sum, local_count = positive_velocity_sum_count(curr_err_per_sample, prev_err_per_sample)
    stats = torch.tensor(
        [local_sum, float(local_count)],
        dtype=torch.float64,
        device=curr_err_per_sample.device,
    )
    if reduce_ddp and distributed_is_ready():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    global_sum = float(stats[0].item())
    global_count = int(stats[1].item())
    global_vel = global_sum / float(global_count) if global_count > 0 else 0.0
    return global_vel, global_sum, global_count


def all_reduce_epoch_totals(values: Sequence[float], device: torch.device, enabled: bool = True) -> List[float]:
    tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
    if enabled and distributed_is_ready():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return [float(x) for x in tensor.tolist()]


def all_reduce_max(value: float, device: torch.device, enabled: bool = True) -> float:
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=device)
    if enabled and distributed_is_ready():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


class EATTrainer:
    """Reusable trainer for epoch-local rollout curriculum with full BPTT.

    Minimal integration:

        eat = EATTrainer(model, optimizer, zero_state, device)
        result = eat.train_epoch(train_loader)
        val = eat.evaluate(val_loader)

    For PDE-Transformer-style class labels, pass `labels_fn`. For custom model
    signatures, pass `forward_fn`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        zero_state: Optional[torch.Tensor],
        device: torch.device | str,
        *,
        config: Optional[EATConfig] = None,
        state: Optional[EATState] = None,
        forward_fn: ForwardFn = default_forward,
        labels_fn: LabelsFn = default_labels,
        scheduler: Optional[Any] = None,
        channels_last: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.zero_state = zero_state
        self.device = torch.device(device)
        self.config = config or EATConfig()
        self.config.validate()
        self.state = state or EATState.from_config(self.config)
        self.forward_fn = forward_fn
        self.labels_fn = labels_fn
        self.scheduler = scheduler
        self.channels_last = channels_last

    def state_dict(self) -> Dict[str, Any]:
        """State to store inside the training checkpoint."""

        return {
            "config": asdict(self.config),
            "state": self.state.state_dict(),
        }

    def load_state_dict(self, state: MutableMapping[str, Any]) -> None:
        """Load EAT state from a checkpoint.

        The config is deliberately not overwritten unless you do that yourself;
        training scripts often change hyperparameters intentionally between
        runs. The checkpointed adaptive promotion threshold is restored.
        """

        if "state" in state:
            self.state.load_state_dict(state["state"])
        else:
            self.state.load_state_dict(state)

    def _labels(self, batch_size: int) -> Optional[torch.Tensor]:
        return self.labels_fn(batch_size, self.device)

    def _autocast_context(self):
        enabled = bool(self.config.use_amp and self.device.type == "cuda")
        if enabled:
            return torch.autocast(device_type="cuda", dtype=self.config.amp_dtype, enabled=True)
        return nullcontext()

    def _predict(self, state: torch.Tensor, labels: Optional[torch.Tensor], training: bool) -> torch.Tensor:
        """Forward pass with optional activation checkpointing."""

        if training and self.config.use_activation_checkpointing:

            def _forward(inp: torch.Tensor, lbl: Optional[torch.Tensor]) -> torch.Tensor:
                with self._autocast_context():
                    out = self.forward_fn(self.model, inp, lbl)
                return out.float()

            if labels is None:

                def _forward_no_labels(inp: torch.Tensor) -> torch.Tensor:
                    with self._autocast_context():
                        out = self.forward_fn(self.model, inp, None)
                    return out.float()

                with suppress(TypeError):
                    return activation_checkpoint(
                        _forward_no_labels,
                        state,
                        use_reentrant=False,
                        preserve_rng_state=self.config.preserve_rng_state,
                    )
                return activation_checkpoint(_forward_no_labels, state, use_reentrant=False)

            with suppress(TypeError):
                return activation_checkpoint(
                    _forward,
                    state,
                    labels,
                    use_reentrant=False,
                    preserve_rng_state=self.config.preserve_rng_state,
                )
            return activation_checkpoint(_forward, state, labels, use_reentrant=False)

        with self._autocast_context():
            pred_raw = self.forward_fn(self.model, state, labels)
        return pred_raw.float()

    def train_batch(
        self,
        batch: Sequence[torch.Tensor],
        stage: EATStage,
        improvement_frac: float,
        step_index: int,
        num_batches: int,
    ) -> EATBatchResult:
        """Train one batch and update the epoch-local curriculum stage."""

        if self.optimizer is None:
            raise ValueError("optimizer is required for train_batch/train_epoch")

        x, y_seq, mask = move_batch_to_device(batch, self.device, channels_last=self.channels_last)
        mask = normalize_mask(mask)
        labels = self._labels(x.shape[0])
        max_rollout = min(int(self.config.max_rollout_len), int(y_seq.shape[1]))
        if max_rollout < 1:
            raise ValueError("y_seq must contain at least one target step")

        effective_frontier_n = min(stage.current_frontier_n, max_rollout - 1)
        rollout_state = x
        prev_err_per_sample = None
        loss_sum = torch.zeros((), device=self.device)
        batch_frontier_vel = None

        for t in range(effective_frontier_n + 1):
            pred_raw = self._predict(rollout_state, labels, training=True)
            y_t = y_seq[:, t]
            loss_sum = loss_sum + F.mse_loss(pred_raw, y_t)

            mse_per_sample = per_sample_mse(pred_raw, y_t).detach()
            if t == effective_frontier_n and t >= 1 and prev_err_per_sample is not None:
                frontier_vel, _, _ = global_positive_velocity(
                    mse_per_sample,
                    prev_err_per_sample,
                    reduce_ddp=self.config.reduce_ddp,
                )
                batch_frontier_vel = float(frontier_vel)

            prev_err_per_sample = mse_per_sample
            if t < effective_frontier_n:
                rollout_state = advance_rollout_state(
                    pred_raw,
                    mask,
                    self.zero_state,
                    apply_mask=self.config.apply_mask_to_rollout,
                )

        loss = loss_sum / float(effective_frontier_n + 1)
        (loss / float(self.config.grad_accum_steps)).backward()

        if (step_index + 1) % self.config.grad_accum_steps == 0 or (step_index + 1) == num_batches:
            if self.config.grad_clip_norm is not None:
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, float(self.config.grad_clip_norm))
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if batch_frontier_vel is not None and effective_frontier_n == stage.current_frontier_n:
            stage.observe_frontier_velocity(batch_frontier_vel, self.config, improvement_frac)

        return EATBatchResult(
            loss=loss.detach(),
            loss_value=float(loss.detach().item()),
            effective_frontier_n=int(effective_frontier_n),
            frontier_velocity=batch_frontier_vel,
            batch_size=int(x.shape[0]),
        )

    def train_epoch(self, loader: Iterable[Sequence[torch.Tensor]], epoch: Optional[int] = None) -> EATEpochResult:
        """Run one training epoch with an epoch-local frontier curriculum."""

        if self.optimizer is None:
            raise ValueError("optimizer is required for train_epoch")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        improvement_frac = float(self.state.promotion_improvement_frac_next)
        stage = EATStage(current_frontier_n=int(self.config.initial_frontier_n))
        n_batches = len(loader) if hasattr(loader, "__len__") else 0

        total_loss_weighted = 0.0
        total_samples = 0
        total_n_weighted = 0.0
        total_n_samples = 0
        frontier_vel_sum = 0.0
        frontier_vel_count = 0
        max_frontier_n = -1
        batch_progress = [] if self.config.collect_batch_frontier_history else None
        batch_n_values = [] if self.config.collect_batch_frontier_history else None

        for step, batch in enumerate(loader):
            result = self.train_batch(batch, stage, improvement_frac, step, max(1, n_batches))
            total_loss_weighted += result.loss_value * result.batch_size
            total_samples += result.batch_size
            total_n_weighted += float(result.effective_frontier_n) * result.batch_size
            total_n_samples += result.batch_size
            max_frontier_n = max(max_frontier_n, result.effective_frontier_n)

            if result.frontier_velocity is not None:
                frontier_vel_sum += float(result.frontier_velocity)
                frontier_vel_count += 1

            if batch_progress is not None and batch_n_values is not None:
                denom = float(max(1, n_batches))
                batch_progress.append(float(step + 1) / denom)
                batch_n_values.append(float(result.effective_frontier_n))

        totals = all_reduce_epoch_totals(
            [
                total_loss_weighted,
                float(total_samples),
                total_n_weighted,
                float(total_n_samples),
                frontier_vel_sum,
                float(frontier_vel_count),
            ],
            self.device,
            enabled=self.config.reduce_ddp,
        )
        max_frontier_n = int(
            all_reduce_max(float(max_frontier_n), self.device, enabled=self.config.reduce_ddp)
        )

        (
            total_loss_weighted,
            total_samples_f,
            total_n_weighted,
            total_n_samples_f,
            frontier_vel_sum,
            frontier_vel_count_f,
        ) = totals

        total_samples = max(1.0, total_samples_f)
        total_n_samples = max(1.0, total_n_samples_f)
        frontier_vel_count = int(frontier_vel_count_f)
        epoch_frontier_vel = (
            frontier_vel_sum / float(frontier_vel_count)
            if frontier_vel_count > 0
            else float("nan")
        )

        if max_frontier_n < self.config.max_rollout_len - 1:
            next_improvement_frac = max(
                float(self.config.min_improvement_frac),
                0.5 * float(improvement_frac),
            )
        else:
            next_improvement_frac = float(improvement_frac)

        if epoch is not None:
            self.state.epoch = int(epoch)
        else:
            self.state.epoch += 1
        self.state.promotion_improvement_frac_next = float(next_improvement_frac)

        if self.scheduler is not None:
            self.scheduler.step()

        return EATEpochResult(
            mse=total_loss_weighted / total_samples,
            avg_target_n=total_n_weighted / total_n_samples,
            train_frontier_vel=epoch_frontier_vel,
            frontier_baseline_vel=stage.reported_baseline_vel(),
            stage_batch_count=int(stage.stage_batch_count),
            promotion_count=int(stage.promotion_count),
            promotion_improvement_frac=float(improvement_frac),
            promotion_improvement_frac_next=float(next_improvement_frac),
            train_frontier_n=int(stage.current_frontier_n),
            max_frontier_n=int(max_frontier_n),
            running_stage_frontier_vel=float(stage.running_stage_vel),
            frontier_vel_count=frontier_vel_count,
            batch_progress=batch_progress,
            batch_n_values=batch_n_values,
        )

    @torch.inference_mode()
    def evaluate(self, loader: Iterable[Sequence[torch.Tensor]], collect_mses: bool = True) -> EATEvalResult:
        """Run full autoregressive validation over all available rollout steps."""

        self.model.eval()
        total_loss_weighted = 0.0
        total_samples = 0
        all_mses: List[np.ndarray] = []
        max_rollout_seen = 0

        for batch in loader:
            x, y_seq, mask = move_batch_to_device(batch, self.device, channels_last=self.channels_last)
            mask = normalize_mask(mask)
            labels = self._labels(x.shape[0])
            max_rollout = min(int(self.config.max_rollout_len), int(y_seq.shape[1]))
            max_rollout_seen = max(max_rollout_seen, max_rollout)
            rollout_state = x
            batch_mses = []
            avg_mse_accum = torch.zeros((), device=self.device)

            for t in range(max_rollout):
                pred_raw = self._predict(rollout_state, labels, training=False)
                y_t = y_seq[:, t]
                mse_loss = F.mse_loss(pred_raw, y_t)
                avg_mse_accum = avg_mse_accum + mse_loss

                if collect_mses:
                    batch_mses.append(per_sample_mse(pred_raw, y_t).cpu().numpy())

                rollout_state = advance_rollout_state(
                    pred_raw,
                    mask,
                    self.zero_state,
                    apply_mask=self.config.apply_mask_to_rollout,
                )

            batch_size = int(x.shape[0])
            total_loss_weighted += float((avg_mse_accum / float(max_rollout)).item()) * batch_size
            total_samples += batch_size
            if collect_mses and batch_mses:
                all_mses.append(np.stack(batch_mses, axis=1))

        totals = all_reduce_epoch_totals(
            [total_loss_weighted, float(total_samples)],
            self.device,
            enabled=self.config.reduce_ddp,
        )
        max_rollout_seen = int(
            all_reduce_max(float(max_rollout_seen), self.device, enabled=self.config.reduce_ddp)
        )
        total_loss_weighted, total_samples_f = totals

        gathered = np.concatenate(all_mses, axis=0) if all_mses else None
        return EATEvalResult(
            mse=total_loss_weighted / max(1.0, total_samples_f),
            all_mses=gathered,
            max_frontier_n=max(0, max_rollout_seen - 1),
        )


def compute_error_dynamics(all_mses: np.ndarray) -> Dict[str, np.ndarray]:
    """Return validation mean MSE and velocity summaries from `[samples, steps]`."""

    if all_mses.ndim != 2:
        raise ValueError("all_mses must have shape [num_samples, rollout_steps]")
    mean_mse = np.mean(all_mses, axis=0)
    velocities = all_mses[:, 1:] - all_mses[:, :-1]
    return {
        "mean_mse": mean_mse,
        "mean_velocity": np.mean(velocities, axis=0),
        "std_velocity": np.std(velocities, axis=0),
    }


def format_epoch_summary(train: EATEpochResult, val: Optional[EATEvalResult] = None) -> str:
    """Compact human-readable summary for logs."""

    baseline = "N/A" if train.frontier_baseline_vel is None else f"{train.frontier_baseline_vel:.6e}"
    stage_vel = (
        "N/A"
        if not np.isfinite(train.running_stage_frontier_vel)
        else f"{train.running_stage_frontier_vel:.6e}"
    )
    val_part = "" if val is None else f"\nval_mse={val.mse:.6e}"
    return (
        f"train_mse={train.mse:.6e}\n"
        f"avg_target_N={train.avg_target_n:.2f} "
        f"current_frontier_N={train.train_frontier_n} "
        f"max_frontier_N={train.max_frontier_n}\n"
        f"stage_frontier_vel={stage_vel} "
        f"epoch_frontier_vel={train.train_frontier_vel:.6e} "
        f"baseline_vel={baseline}\n"
        f"stage_batches={train.stage_batch_count} "
        f"promotions={train.promotion_count} "
        f"required_improvement={100.0 * train.promotion_improvement_frac:.2f}% "
        f"next_epoch={100.0 * train.promotion_improvement_frac_next:.2f}%"
        f"{val_part}"
    )
