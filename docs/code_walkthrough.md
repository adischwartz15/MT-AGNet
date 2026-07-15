# Code Walkthrough

This document traces one end-to-end flow through the repository: one
training sample. File references link to the actual source.

## One Training Sample

### Flow overview

```
raw metadata → validation → split assignment → Dataset.__getitem__
→ image transform → DataLoader batch → model.forward → backbone
→ shared embedding → task adapters → task heads → age quantile loss
→ classification loss → multi-task loss balancing → backward pass
→ optimizer update → scheduler → validation metrics → checkpoint
→ history/plots
```

### Tensor shape table (default architecture: shared_adapters, custom_resnet18)

| Stage | Tensor shape | Meaning |
|---|---|---|
| Input batch | `[B, 3, 128, 128]` | RGB face images (B=64 default) |
| After stem (conv7x7 + pool) | `[B, 64, 32, 32]` | Stem features |
| After layer1 | `[B, 64, 32, 32]` | First residual stage |
| After layer2 | `[B, 128, 16, 16]` | Second residual stage (stride 2) |
| After layer3 | `[B, 256, 8, 8]` | Third residual stage (stride 2) |
| After layer4 | `[B, 512, 4, 4]` | Final residual stage (stride 2) |
| After avgpool + flatten | `[B, 512]` | Global average pooled embedding |
| embedding_proj (Identity for 512→512) | `[B, 512]` | Shared embedding |
| Age adapter output | `[B, 512]` | Age-specific representation |
| Gender adapter output | `[B, 512]` | Gender-label-specific representation |
| Age head trunk output | `[B, 128]` | Age hidden features |
| center_raw, lower_delta, upper_delta | `[B]` each | Raw head outputs |
| q10, q50, q90 (clamped) | `[B]` each | Age quantile predictions |
| q10_raw, q50_raw, q90_raw | `[B]` each | Unclamped quantiles (for loss) |
| Gender head trunk output | `[B, 128]` | Gender hidden features |
| Gender logits | `[B, 2]` | Classification logits |

### Step-by-step trace

#### 1. Raw metadata → validation
- **File**: `src/data/metadata.py` — Parses UTKFace filenames (`age_gender_race_date.jpg`)
- **File**: `src/data/validation.py` — Checks min image size, file size, detects duplicates (hash + path), drops corrupt images
- **Why**: Ensures data quality before any model sees the images

#### 2. Split assignment
- **File**: `src/data/split_utils.py` — 4-way stratified split: train (60%), validation (15%), calibration (10%), test (15%)
- **Config**: `configs/data.yaml` `split.*` section, seed=42
- **Why labels use masks**: A sample may have age only, gender only, both, or neither. The mask approach (`age_mask`, `gender_mask`) lets both tasks share the same batch without requiring all labels present.

#### 3. `Dataset.__getitem__`
- **File**: `src/data/dataset.py`, class `FaceMultiTaskDataset`
- **Input**: DataFrame row index
- **Output**: `dict` with keys `image` (tensor), `age` (float), `age_mask` (bool), `gender_label` (long), `gender_mask` (bool), `index` (int)
- **Gradients**: No (data loading)

#### 4. Image transform (training)
- **File**: `src/data/transforms.py`, class `TrainTransform`
- **Operations**: `random_crop_resize` → `random_horizontal_flip` → `color_jitter` → `to_tensor` → `normalize`
- **Normalization**: ImageNet mean/std `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` — standard constants, not pretrained weights
- **Output shape**: `[3, 128, 128]` float32 tensor

#### 5. DataLoader batch
- **File**: `src/training/trainer.py`, `Trainer.__init__` L163-171
- **batch_size**: 64 (from `configs/training.yaml`)
- **shuffle**: True for train, False for val
- **worker_init_fn**: `seed_worker` for reproducible augmentation across workers
- **pin_memory**: True only when device is CUDA

#### 6. Model forward (`MultiTaskFaceModel.forward`)
- **File**: `src/models/multitask_model.py` L156-164

#### 7. Backbone (`CustomResNet18.forward`)
- **File**: `src/models/custom_resnet.py` L153-157
- Calls `forward_features` → `avgpool` → `flatten` → `embedding_proj`
- **Gradients**: Yes (all layers trainable in warm-up stage)
- **Config**: `configs/model.yaml` `model.backbone.*`

#### 8. Shared embedding → Task adapters
- **File**: `src/models/adapters.py`, class `BottleneckAdapter`
- **Equation**: `adapter_output = z + up(dropout(gelu(down(z))))`
- **Why adapters are residual**: The `z + delta` form means the adapter starts as near-identity (up_proj initialized to zero), so early training doesn't disturb the shared representation
- **Why zero-initialize up_proj**: `nn.init.zeros_(self.up_proj.weight)` and `nn.init.zeros_(self.up_proj.bias)` make the adapter a no-op at initialization. The shared backbone representation passes through unchanged initially; adapters only diverge as they learn task-specific adjustments.

#### 9. Task heads
- **Age head** (`src/models/heads.py`, `AgeQuantileHead`): trunk(Linear→GELU→Dropout) → three separate linear heads for center, lower_delta, upper_delta
  - `q50 = age_min + sigmoid(center_raw) * (age_max - age_min)` — sigmoid constrains to [age_min, age_max]
  - `q10 = q50 - softplus(lower_delta)` — softplus guarantees non-negative
  - `q90 = q50 + softplus(upper_delta)` — guarantees q10 ≤ q50 ≤ q90
  - **Why raw unclamped quantiles in loss**: Clamping can zero out gradients at boundaries; the loss uses `q10_raw/q50_raw/q90_raw` so gradients always flow
  - **Why display outputs may be clamped**: Clamped versions for user-facing display prevent showing impossible ages like -5
- **Gender head** (`src/models/heads.py`, `GenderClassificationHead`): trunk(Linear→GELU→Dropout) → Linear(hidden, num_classes) → raw logits
  - **Why logits not softmax**: `F.cross_entropy` expects raw logits and applies log_softmax internally for numerical stability
  - **Why not apply softmax before cross-entropy**: Would compute `log(softmax(x))` which is numerically less stable than `log_softmax(x)` which `cross_entropy` uses

#### 10. Age quantile loss (pinball loss)
- **File**: `src/losses/quantile_loss.py`
- `L_tau(y, yhat) = max(tau * (y - yhat), (tau - 1) * (y - yhat))`
- Mean of three quantile losses (q10 at τ=0.10, q50 at τ=0.50, q90 at τ=0.90)
- Masked: only samples with `age_mask=True` contribute; denominator = count of valid samples
- If all masked out: returns `losses.sum() * 0.0` (zero with gradient)

#### 11. Classification loss
- **File**: `src/losses/multitask_loss.py` L64-69
- `F.cross_entropy` with optional class weights, per-sample, then masked mean
- Gender mask handles missing labels same way as age

#### 12. Multi-task loss balancing
- **File**: `src/losses/multitask_loss.py` L78-95
- **Fixed mode**: `total = age_weight * age_loss + gender_weight * gender_loss`
- **Learned uncertainty mode**: `total = exp(-s_age) * age_loss + s_age + exp(-s_gender) * gender_loss + s_gender`
  - **Why `+ log_var`**: Acts as a regularizer — prevents the model from learning to ignore a task by setting its precision to zero (which would set the loss weight to zero but the `+ s` term penalizes that)
  - **Why task term omitted when no labels**: A loss of 0 combined with the `+ s` bias term would contribute meaningless regularization with no supervisory signal
- **Warmup**: First N epochs use fixed weights even in learned mode (configured via `loss_balancing.learned_uncertainty.warmup_epochs`)

#### 13. Backward pass
- **File**: `src/training/trainer.py` L246-261
- `scaler.scale(loss).backward()` — AMP gradient scaling
- `scaler.unscale_(optimizer)` — unscale before clipping
- `clip_grad_norm_(model.parameters(), grad_clip_norm)` — gradient clipping (default 1.0)
- **Why gradient clipping exists**: Prevents exploding gradients, especially important in multi-task training where conflicting task gradients can create large combined updates
- **Why mixed precision requires GradScaler**: FP16 has limited dynamic range; the scaler prevents underflow/overflow in gradients

#### 14. Optimizer update
- `scaler.step(optimizer)` — may skip if inf/NaN gradient detected
- `scaler.update()` — adjusts scale factor
- **Why different learning rates may be used**: Differential LR (`training.differential_lr`) gives backbone a smaller LR than adapters/heads, protecting pretrained or jointly-optimized features

#### 15. Scheduler
- **File**: `src/training/trainer.py` L59-79
- Linear warmup (1 epoch by default) → cosine annealing
- Only steps when optimizer actually stepped (tracked via `scale_before_step`)
- **Why validation does not update weights**: `optimizer is None` in val loop, `torch.set_grad_enabled(False)`, `model.eval()` set

#### 16. Checkpoint selection
- **File**: `src/training/checkpointing.py`
- Three trackers: best `age_mae` (min), best `gender_accuracy` (max), best `balanced_score` (max)
- `balanced_score = gender_accuracy - (age_mae / age_max)` — normalized joint metric
- **Why calibration is not part of the neural forward pass**: Calibration is a post-hoc statistical adjustment applied to model outputs, not a learnable component — it uses a separate data split
