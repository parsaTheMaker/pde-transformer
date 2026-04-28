# EAT: Error-Accelaration Aware Training

`EAT.py` is a reusable PyTorch module for training autoregressive neural
simulators with an epoch-local rollout curriculum.

EAT stands for **Error-Accelaration Aware Training**. The central idea is to
train an autoregressive model through a gradually expanding rollout frontier
`N`, while using the model's own rollout error dynamics to decide when that
frontier should move forward.

## Core Method

The module implements the following training behavior:

1. **Raw-output MSE loss**

   The loss is computed directly between the raw model prediction and the target:

   ```python
   loss_t = mse(pred_raw_t, y_t)
   ```

   The obstacle/domain mask is not applied to the loss.

2. **Mask only for autoregressive state advancement**

   After a prediction is made, the next input state is formed with:

   ```python
   state = torch.lerp(zero_state, pred_raw, mask)
   ```

   This means masked-out cells are forced to the normalized zero state only when
   the rollout advances.

3. **Full BPTT through steps `0..N`**

   For the current frontier `N`, the batch loss is the equal-weight average:

   ```python
   loss = mean([mse(pred_0, y_0), ..., mse(pred_N, y_N)])
   ```

   The graph is kept through the autoregressive chain, so gradients flow through
   every step from `0` to `N`.

4. **Epoch-local curriculum**

   Every epoch starts from `N=1` by default. Promotion inside the epoch can move
   the frontier to `N=2`, `N=3`, and so on, up to `max_rollout_len - 1`.

5. **Promotion from positive error velocity**

   For each frontier-reaching batch, EAT computes per-sample MSE at the current
   frontier and compares it to the previous step:

   ```python
   velocity = E_N - E_{N-1}
   ```

   Only positive velocities are averaged. This focuses the signal on samples
   where the rollout is getting worse.

6. **Stage-based promotion rule**

   For the current frontier stage, EAT waits for enough frontier-reaching
   batches. It then compares:

   - baseline: mean velocity over the older stage batches
   - recent: mean velocity over the latest `recent_promo_window` batches

   Promotion happens when:

   ```text
   recent <= baseline * (1 - promotion_improvement_frac)
   ```

7. **Persisted adaptive promotion threshold**

   If an epoch does not reach the full rollout frontier, the next epoch's
   required improvement is halved, bounded below by `min_improvement_frac`.
   This value is checkpointable through `EATState`.

## Tensor Contract

EAT expects batches shaped like:

```text
x      [B, C, H, W]
y_seq  [B, T, C, H, W]
mask   [B, 1, H, W] or [B, H, W] optional
```

`x` and `y_seq` should already be normalized in the same space your model uses.
If your physical domain has an obstacle mask or valid-fluid mask, pass it as
`mask`. If your model has no mask, batches may be `(x, y_seq)` and EAT will use
the raw prediction as the next state.

For the Karman scripts in this repo:

```python
zero_state = ((torch.zeros(3, device=device) - mean.to(device)) / std.to(device)).view(1, 3, 1, 1)
```

That `zero_state` is what fills masked cells during rollout advancement.

## Minimal Use

```python
import torch
from EAT import EATConfig, EATTrainer, format_epoch_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = EATConfig(
    max_rollout_len=8,
    initial_frontier_n=1,
    initial_improvement_frac=0.30,
    min_improvement_frac=0.01,
    min_stage_batches=30,
    recent_promo_window=10,
    grad_accum_steps=1,
    grad_clip_norm=1.0,
    use_amp=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    use_activation_checkpointing=True,
)

eat = EATTrainer(
    model=model,
    optimizer=optimizer,
    zero_state=zero_state,
    device=device,
    config=config,
)

for epoch in range(1, epochs + 1):
    train_result = eat.train_epoch(train_loader, epoch=epoch)
    val_result = eat.evaluate(val_loader)
    print(format_epoch_summary(train_result, val_result))
```

By default, `EATTrainer` calls:

```python
out = model(state)
pred = out.sample if hasattr(out, "sample") else out
```

This works for plain PyTorch models and PDE-Transformer-style outputs.

## PDE-Transformer Class Labels

If your model uses task labels or class labels, pass them with a small label
provider:

```python
task_label = torch.tensor([1000], dtype=torch.long, device=device)

def labels_fn(batch_size, device):
    return task_label.expand(batch_size)

eat = EATTrainer(
    model=model,
    optimizer=optimizer,
    zero_state=zero_state,
    device=device,
    config=config,
    labels_fn=labels_fn,
)
```

The default forward function will call:

```python
model(state, class_labels=labels)
```

and then read `.sample` if the model returns a diffusers-style object.

## Custom Model Signatures

If your model uses a different forward signature, pass `forward_fn`:

```python
def forward_fn(model, state, labels):
    del labels
    return model.rollout_step(state, dt=0.01)

eat = EATTrainer(
    model=model,
    optimizer=optimizer,
    zero_state=None,
    device=device,
    config=EATConfig(apply_mask_to_rollout=False),
    forward_fn=forward_fn,
)
```

Return either a tensor or an object with a `.sample` tensor.

## Checkpointing

Store EAT state in your usual training checkpoint:

```python
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eat": eat.state_dict(),
    },
    "last.ckpt",
)
```

Load it on resume:

```python
ckpt = torch.load("last.ckpt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
eat.load_state_dict(ckpt["eat"])
```

The checkpointed field that matters most is:

```text
promotion_improvement_frac_next
```

This is the next epoch's required improvement threshold.

## Configuration Reference

| Training concept | EAT.py setting |
| --- | --- |
| Maximum rollout length | `EATConfig(max_rollout_len=8)` |
| Initial required frontier improvement | `initial_improvement_frac=0.30` |
| Minimum required frontier improvement | `min_improvement_frac=0.01` |
| Minimum baseline batches per stage | `min_stage_batches=30` |
| Recent promotion comparison window | `recent_promo_window=10` |
| Gradient accumulation | `grad_accum_steps` |
| Gradient clipping | `grad_clip_norm` |
| Autocast mixed precision | `use_amp` |
| Activation checkpointing | `use_activation_checkpointing` |
| Custom model prediction | `forward_fn` |
| Per-sample rollout error | `EAT.per_sample_mse()` |
| Positive error velocity | `EAT.global_positive_velocity()` |
| Training epoch | `EATTrainer.train_epoch()` |
| Validation rollout | `EATTrainer.evaluate()` |

Your training script remains responsible for dataset construction,
normalization, model construction, optimizer setup, DDP wrapping, plotting,
videos, and experiment logs. `EAT.py` owns only the rollout-curriculum training
methodology.

## Method Details

### Rollout Loss

At frontier `N`, EAT performs:

```python
state = x
loss_sum = 0

for t in range(N + 1):
    pred_raw = model(state)
    loss_sum += mse(pred_raw, y_seq[:, t])

    if t < N:
        state = lerp(zero_state, pred_raw, mask)

loss = loss_sum / (N + 1)
loss.backward()
```

The equal weighting matters. Later steps are harder, but the method does not
upweight them directly. It lets the curriculum decide when the model is ready
for longer horizons.

### Error Velocity

For each sample:

```text
E_t = mse(pred_t, y_t)
v_t = E_t - E_{t-1}
```

EAT averages only `v_t > 0`. If a sample improves or stays flat, it does not
raise the frontier velocity statistic.

This makes the promotion signal answer a specific question:

> Among the samples where error is still growing, is that growth now lower than
> it was earlier in this frontier stage?

### Frontier Promotion

Within an epoch, each frontier value `N` is a stage. A stage collects frontier
velocity values from batches that actually trained at that frontier.

After at least:

```text
min_stage_batches + recent_promo_window
```

frontier-reaching batches, EAT computes:

```python
baseline = mean(stage_velocities[:-recent_promo_window])
recent = mean(stage_velocities[-recent_promo_window:])
```

Then:

```python
if recent <= baseline * (1 - improvement_frac):
    N += 1
    reset_stage_statistics()
```

The stage reset is important. Each new frontier has its own baseline and recent
window.

### Next-Epoch Improvement Threshold

At the end of an epoch:

```python
if max_frontier_n < max_rollout_len - 1:
    next_improvement = max(min_improvement_frac, 0.5 * current_improvement)
else:
    next_improvement = current_improvement
```

This is how EAT adapts without a separate calibration or survey pass.
If the curriculum cannot reach the full horizon, the next epoch asks for a less
strict improvement before promotion.

## DDP Notes

`EAT.py` can all-reduce scalar totals and frontier velocity statistics when
`torch.distributed` is initialized:

```python
config = EATConfig(reduce_ddp=True)
```

This makes the positive velocity statistic global across ranks. Wrap the model
in DDP outside EAT, just as you normally would in a training script.

If you need gathered validation per-sample MSE arrays across ranks for plotting,
keep your existing gather utility. `EAT.evaluate()` returns local `all_mses`;
the scalar validation MSE is reduced.

## Practical Settings

For a stable default setup, start with:

```python
EATConfig(
    max_rollout_len=8,
    initial_frontier_n=1,
    initial_improvement_frac=0.30,
    min_improvement_frac=0.01,
    min_stage_batches=30,
    recent_promo_window=10,
    grad_accum_steps=1,
    grad_clip_norm=1.0,
    use_amp=True,
    use_activation_checkpointing=True,
)
```

For a smaller dataset or faster debugging:

```python
EATConfig(
    max_rollout_len=4,
    min_stage_batches=5,
    recent_promo_window=3,
    use_activation_checkpointing=False,
)
```

For a model without masks:

```python
EATConfig(apply_mask_to_rollout=False)
```

and pass `zero_state=None`.

## Common Integration Mistakes

1. **Applying the mask to the loss**

   Do not mask `pred_raw` before MSE if you want to reproduce the reference
   methodology. Mask only the next autoregressive state.

2. **Detaching the rollout state during training**

   EAT is full BPTT through `0..N`. Detaching between steps changes the method
   into truncated or one-step training.

3. **Starting the epoch from the previous epoch's final N**

   The reference curriculum is epoch-local. Each epoch restarts at `N=1`; only
   the promotion threshold persists.

4. **Using target step count instead of frontier index**

   `N=1` means two supervised predictions: step `0` and step `1`.

5. **Forgetting normalized zero state**

   If your data is normalized, `zero_state` must be zero in physical units
   transformed into normalized units, not simply `torch.zeros_like(x)`.

## Reading the Results

`train_epoch()` returns `EATEpochResult`:

```python
result.mse
result.avg_target_n
result.train_frontier_vel
result.frontier_baseline_vel
result.stage_batch_count
result.promotion_count
result.promotion_improvement_frac
result.promotion_improvement_frac_next
result.train_frontier_n
result.max_frontier_n
result.running_stage_frontier_vel
result.batch_progress
result.batch_n_values
```

The most useful signals are:

- `max_frontier_n`: the farthest frontier reached this epoch
- `promotion_count`: how many times the curriculum advanced
- `promotion_improvement_frac_next`: what will be used next epoch
- `running_stage_frontier_vel`: current stage's mean positive error velocity
- `avg_target_n`: average frontier used across samples

Use:

```python
from EAT import format_epoch_summary
print(format_epoch_summary(train_result, val_result))
```

for a compact log line.

## Validation Error Dynamics

`evaluate(..., collect_mses=True)` returns per-sample, per-step MSE in
`val_result.all_mses` on the local rank:

```python
from EAT import compute_error_dynamics

dynamics = compute_error_dynamics(val_result.all_mses)
mean_mse = dynamics["mean_mse"]
mean_velocity = dynamics["mean_velocity"]
std_velocity = dynamics["std_velocity"]
```

These arrays can be used to build validation error dynamics tables or plots.
