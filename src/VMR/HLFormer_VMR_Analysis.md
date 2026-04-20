# HLFormer-VMR: Architecture & Innovations Analysis

> **Codebase:** `ICCV25-HLFormer/src/VMR/` (v32 — Charades-STA configuration)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [HLFormerBlock: Hybrid Euclidean-Hyperbolic Encoder](#2-hlformerblock-hybrid-euclidean-hyperbolic-encoder)
3. [Content-Adaptive Query Initialization (CAQ)](#3-content-adaptive-query-initialization-caq)
4. [Boundary Refinement Head](#4-boundary-refinement-head)
5. [Hyperbolic Geometry Components](#5-hyperbolic-geometry-components)
6. [Loss Functions](#6-loss-functions)
7. [Training Strategy](#7-training-strategy)
8. [Data Processing](#8-data-processing)
9. [Summary: Key Innovations](#9-summary-key-innovations)
10. [Hyperparameter Reference](#10-hyperparameter-reference)

---

## 1. Architecture Overview

### 1.1 Processing Pipeline

```
Multi-stream Video Projection (SlowFast + CLIP + BLIP → H=384)
  → Text Self-Attention Encoder
  → Early Query Fusion (T2V with global token, 6 layers)
  → HLFormerBlock (8-branch Euclidean + Hyperbolic encoder)
  → Decoder (CAQ init + 6-layer iterative refinement)
  → Prediction Heads (quality score + span)
  → Boundary Refinement Head (Gaussian pooling + MLP, activated at epoch 15)
  → Saliency Heads (Euclidean + Hyperbolic)
  → Hyperbolic Entailment Head
```

**Source:** `Models/vmr_model.py` — class `HLFormer_VMR`

### 1.2 Multi-Stream Video Projection

Three heterogeneous feature streams are independently projected then fused:

```
SlowFast (2304-D) → Linear(2304, H) + LayerNorm + ReLU
CLIP     (512-D)  → Linear(512,  H) + LayerNorm + ReLU
BLIP     (768-D)  → Linear(768,  H) + LayerNorm + ReLU
                   → Concat(3 × H) → Linear(3H, H) + LayerNorm + GELU → (B, L_v, H)
```

**Source:** `Models/vmr_model.py` — class `MultiStreamVidProjection`

### 1.3 Text Encoder

Text tokens are encoded via self-attention with trainable positional encoding:

```
src_txt: (B, L_t, 1280)  →  Linear(1280, H)
  → TrainablePositionalEncoding
  → EuclideanAttentionBlock (self-attention + FFN)
  → LayerNorm + Dropout
  → txt_feat: (B, L_t, H)
```

### 1.4 Early Query Fusion (T2V Encoder)

Cross-modal fusion where video queries attend to text (6 stacked layers):

```
Input:  [global_token | video] as query,  text as key/value
For each T2V layer:
    output = cross_attn(q=[global | video], k=text, v=text)
    output = residual + FFN

Extract:
    global_rep = output[:, 0, :]    →  (B, H)    cross-modal summary
    vid_feat   = output[:, 1:, :]   →  (B, L_v, H)  text-conditioned video
```

**Source:** `Models/vmr_model.py` — class `T2VEncoderLayer`, method `_early_query_fusion`

### 1.5 Feature Noise Augmentation (v32)

After video projection and text encoding, Gaussian noise is injected during training:

```python
# Applied after input_vid_proj + vid_pos_embed and _encode_query
if training and feat_noise_std > 0:
    vid_feat_proj += randn_like(vid_feat_proj) * feat_noise_std   # σ=0.05
    txt_feat      += randn_like(txt_feat)      * feat_noise_std
```

Prevents the model from memorizing exact projected feature patterns. Disabled at inference.

**Source:** `Models/vmr_model.py` — `HLFormer_VMR.forward`, step 2b

### 1.6 Decoder with Iterative Refinement

The decoder uses sine-conditioned cross-attention with per-layer reference point refinement:

```
For each of 6 decoder layers:
    1. Generate sine embeddings from current reference points (center, width)
    2. Self-attention: queries attend to each other
    3. Cross-attention:
         Q = ca_qcontent(tgt) + ca_qpos_sine(sine_embed)
         K = ca_kcontent(memory)     # memory = [vid_feat | txt_feat]
         V = ca_v(memory)
    4. FFN
    5. Predict delta: Δ = bbox_embed[i](output)
    6. Refine: ref_new = sigmoid(inverse_sigmoid(ref_old) + Δ)
    7. Save intermediate output for auxiliary loss
```

**Source:** `Models/vmr_model.py` — classes `VMRDecoder`, `VMRDecoderLayer`

---

## 2. HLFormerBlock: Hybrid Euclidean-Hyperbolic Encoder

The central architectural innovation. 8 parallel attention branches (4 Euclidean + 4 Hyperbolic) with per-position learned fusion.

**Source:** `Models/HLFormer/model_components.py` — class `HLFormerBlock`

### 2.1 Branch Structure

With `attention_num=8`: `num_block = 8 // 2 - 1 = 3`, producing 4 branches per geometry:

| Branch | Type | Window Width | Receptive Field |
|--------|------|-------------|-----------------|
| E0 | Euclidean Gaussian | None (global) | Full sequence |
| E1 | Euclidean Gaussian | wid=2 | Local |
| E2 | Euclidean Gaussian | wid=4 | Medium |
| E3 | Euclidean Gaussian | wid=8 | Wide |
| H0 | Hyperbolic Lorentz | None (global) | Full sequence |
| H1 | Hyperbolic Lorentz | wid=2 | Local |
| H2 | Hyperbolic Lorentz | wid=4 | Medium |
| H3 | Hyperbolic Lorentz | wid=8 | Wide |

Each Euclidean branch: `EuclideanAttentionBlock` = multi-head self-attention (with optional Gaussian window mask) + FFN + LayerNorm + residual.

Each Hyperbolic branch: `LorentzAttentionBlock` = project to hyperboloid via `exp_map0` → Lorentzian multi-head attention → `log_map0` back to Euclidean.

### 2.2 Gaussian Window Mask

For branches with `wid ≠ None`, attention scores are element-wise multiplied by a 1D Gaussian:

```python
center = arange(L) / L                         # (L,)
σ      = wid / 9                                # fixed σ from window width
mask   = (0.3989 / σ) × exp(−(pos − center)² / (2σ²))
mask   = mask / mask.max(dim=-1)                # normalize to [0, 1]
attn_scores = attn_scores * mask                # apply before softmax
```

**Source:** `Models/HLFormer/model_components.py` — `EuclideanGaussianAttention.generate_gauss_weight`

### 2.3 Per-Position Learned Fusion

```python
# Run all 8 branches independently
outputs = [branch(input) for branch in euclidean_branches + hyperbolic_branches]
oo = torch.cat([o.unsqueeze(-1) for o in outputs], dim=-1)  # (B, L_v, H, 8)

# Compute fusion weights using cross-attention
weight_token = global_rep  # from T2V encoder (B, 1, H)
for i in range(8):
    weight[i] = CrossAttention(query=weight_token, kv=oo[..., i])

weight = Linear(H → L_v) → ReLU → Dropout → Linear
weight = softmax(weight / sft_factor, dim=-1)   # sft_factor=0.1 (Charades)
output = sum(oo * weight)                        # (B, L_v, H)
```

The `sft_factor=0.1` controls softmax temperature — lower values produce sharper branch selection (specialization per position).

---

## 3. Content-Adaptive Query Initialization (CAQ)

Decoder queries are conditioned on both text semantics and video content before refinement begins.

**Source:** `Models/vmr_model.py` — `HLFormer_VMR.forward`, lines 761–821

```python
# 1. Text-gated seeds
txt_mean  = masked_average(txt_feat)                    # (B, H)
txt_gate  = caq_txt_gate(txt_mean)                      # (B, Q×H) → (B, Q, H)
q_seeds   = learned_query_weight + txt_gate             # text-conditioned

# 2. Cross-attention over video
caq_out   = cross_attn(q_seeds, vid_feat, vid_feat)     # (B, Q, H)
caq_out   = LayerNorm(caq_out)

# 3. Fuse with base embedding
tgt = base_query_embed + caq_out                        # (B, Q, H)

# 4. Adaptive reference points
ref_offset = caq_ref_head(caq_out)                      # (B, Q, 2)
ref = sigmoid(inverse_sigmoid(base_ref) + ref_offset)   # data-dependent init
```

Queries start aligned with both text content and relevant video locations, enabling faster convergence compared to fixed learned embeddings.

---

## 4. Boundary Refinement Head

Post-decoder local boundary refinement using Gaussian soft-attention pooling.

**Source:** `Models/vmr_model.py` — class `BoundaryRefinementHead`

```python
# For each predicted span (center, width) → (start, end):
for boundary in [start, end]:
    # Gaussian soft-attention around predicted boundary
    weights = gaussian(positions, boundary, σ)    # σ learnable per boundary
    pooled  = weighted_avg(vid_feat, weights)      # local context features

# Concatenate start/end pooled features
joint = concat(pooled_start, pooled_end)           # (B, Q, 2H)
delta = joint_mlp(joint)                           # (B, Q, 2) — predicted correction
refined_span = clamp(span + delta, ±max_delta)     # max_delta=0.25 (v32: ±19 frames)
```

Configuration (v32):
- `boundary_refine_window = 15`: σ_init = 15/(2×75) = 0.1 normalized → ~7.5 frames std dev *(v31: 10 frames)*
- `boundary_refine_learnable_sigma = True`: σ learned per boundary
- `boundary_refine_max_delta = 0.25`: ±19 frames correction range *(v31: 0.15 → ±11 frames — too restrictive)*
- Zero-initialized output layer: starts as identity, gradually activates
- **Delayed activation**: disabled until epoch 15 via loss schedule *(v31: active from epoch 0, caused `loss_giou_refined > loss_giou`)*

> **v32 rationale:** In v31 logs, `loss_giou_refined` (0.44) was consistently higher than `loss_giou` (0.35), meaning the refinement head was making predictions *worse* than the base decoder. This was caused by activating it before the decoder converged. Delaying to epoch 15 and widening `max_delta` to 0.25 fixes both issues.

---

## 5. Hyperbolic Geometry Components

### 5.1 Lorentz Model

Hyperbolic space (negative curvature) naturally models hierarchical and multi-scale temporal structures. HLFormer uses the Lorentz hyperboloid model for numerical stability.

**Source:** `Models/HLFormer/lorentz.py`, `Models/onmt/lorentz.py`

Key operations:
- **exp_map0(x, κ)**: Euclidean → hyperboloid projection
- **log_map0(x, κ)**: Hyperboloid → Euclidean (tangent space)
- **Lorentzian inner product**: `⟨x, y⟩_L = x₀y₀ − Σ xᵢyᵢ` (Minkowski metric)
- **pairwise_dist**: Hyperbolic distance via `acosh(−⟨x, y⟩_L)`
- **mid_point**: Weighted Fréchet mean on the hyperboloid (replaces Euclidean weighted sum)

### 5.2 Curvature Parameter

```python
curv = nn.Parameter(tensor(curv_init))       # log-space: κ = exp(curv)
κ = curv.exp()                                # actual curvature
κ = clamp(κ, κ_init/10, κ_init×10)          # prevent divergence
```

- Charades: `curv_init=1.0`, `learn_curv=False` (fixed κ ≈ 2.718)
- QVHighlights: `curv_init=1.0`, `learn_curv=True`

### 5.3 Lorentzian Multi-Head Attention

Used in the 4 Hyperbolic branches of HLFormerBlock:

**Source:** `Models/HLFormer/hyper_nets.py` — class `LorentzMultiHeadedAttention`

```python
# Project to hyperboloid
Q = exp_map0(LorentzLinear(query))
K = exp_map0(LorentzLinear(key))
V = exp_map0(LorentzLinear(value))

# Attention via Lorentzian inner product (not Euclidean dot product)
scores = (2 + 2 × cinner(Q, K)) / scale        # cinner = Lorentzian IP
attn_weights = softmax(scores + optional_gaussian_mask)

# Aggregation via Fréchet mean (not Euclidean weighted sum)
output = mid_point(V, attn_weights)              # weighted mean on hyperboloid
output = log_map0(output)                        # project back to tangent space
```

### 5.4 Alpha Scaling

Features are scaled before hyperboloid projection to control norm on the manifold:

```python
video_alpha   = nn.Parameter(log(1 / sqrt(H)))   # ≈ −2.95 for H=384
textual_alpha = nn.Parameter(log(1 / sqrt(H)))

# Constrained to ≤ 0 (exp(alpha) ≤ 1) via gradient hooks to prevent norm explosion
v_scaled = vid_global × exp(video_alpha)
hyp_vid  = exp_map0(v_scaled, κ)
```

### 5.5 Hyperbolic Entailment Head

Three-component entailment loss using cone geometry:

```python
# Global video/text representations on hyperboloid
hyp_vid = exp_map0(attention_pool(vid_feat) × exp(video_alpha), κ)
hyp_txt = exp_map0(masked_avg(txt_feat) × exp(textual_alpha), κ)

# Entailment: text must lie inside video's cone
angle    = oxy_angle(hyp_vid, hyp_txt, κ)     # angle between them
aperture = half_aperture(hyp_vid, κ)           # video's cone half-angle

loss_pos = clamp(angle − aperture + margin, min=0)    # text inside cone
loss_neg = clamp(aperture − neg_angle + margin, min=0) # wrong text outside
```

Hard negative mining: closest-in-batch negatives (minimum hyperbolic distance) rather than trivial cyclic shift.

Span-level entailment: Hungarian-matched span features are also constrained to entail query text, providing a direct geometric signal for boundary precision.

### 5.6 Hyperbolic Saliency

Frame-level saliency via Lorentzian inner product:

```python
hyp_saliency[t] = −⟨hyp_frame[t], hyp_text⟩_L
# Margin ranking: hyp_sal[pos_frame] > hyp_sal[neg_frame] + margin
```

Provides a geometry-aware saliency channel alongside the Euclidean saliency head.

---

## 6. Loss Functions

**Source:** `Losses/vmr_loss.py` — class `VMRSetCriterion`

### 6.1 Span Localization (Coarse)

Combined loss on decoder span predictions:

```python
loss_span = L1(pred_cxw, gt_cxw)                       # center-width regression

loss_diou  = DIoU(pred_xx, gt_xx)                       # center-distance penalty
loss_alpha = 1 − IoU(pred_xx, gt_xx)^α                  # α=2.5 amplifies medium-IoU gradient
loss_giou  = 0.5 × loss_diou + 0.5 × loss_alpha        # combined IoU loss
```

- **DIoU** over GIoU: provides center-alignment gradient even when spans are non-overlapping
- **Alpha-IoU** (`α=2.5`): amplifies gradient in the 0.5–0.7 IoU zone, directly targeting the R1@0.7 metric
- **Boundary smooth-L1** (`β=0.05`): direct (start, end) supervision (currently disabled; boundary supervision handled by refinement head)

### 6.2 Span Localization (Refined)

Applied to `BoundaryRefinementHead` output using the same Hungarian indices. **Activated at epoch 15 in v32** (was epoch 0 in v31):

```python
loss_boundary_refined = smooth_l1(pred_xx, gt_xx, β=0.05)
loss_giou_refined     = alpha_iou(pred_xx, gt_xx, α=2.5)   # Alpha-IoU only (no DIoU)
```

### 6.3 Quality Score Classification

Soft target formulation replacing binary foreground/background:

```python
soft_target[matched]   = IoU_with_gt.clamp(min=0.1)     # [0.1, 1.0]
soft_target[unmatched] = 0.0

pos_weight = min((n_total − n_fg) / n_fg, 25)           # adaptive class balancing
loss_label = BCE_with_logits(pred_logits, soft_target, pos_weight=pos_weight)
```

At inference: `sigmoid(pred_logits) ≈ expected localization quality` — quality-aware detection rather than binary presence/absence.

### 6.4 Saliency Loss (3 Components)

```python
# 1. Negative-pair suppression: mismatched text → low saliency
loss_neg = −log(1 − sigmoid(scores_neg)) × vid_mask

# 2. Multi-level ranking contrastive (11 rank thresholds)
for rank_thr in 1..11:
    loss_rank += cross_entropy over pos/neg concatenation
loss_rank /= n_active_thresholds    # fixed: divide by actual active count, not hardcoded 11

# 3. Margin-based ranking
loss_margin = clamp(margin + neg_score − pos_score, min=0)

loss_saliency = loss_margin + loss_rank + loss_neg
```

Negative saliency (v31): simplified to shifted dot product instead of full encoder re-run. The negative pair only needs "mismatched text → low saliency" — no need to recompute video features. Saves ~40% forward-pass time.

**v32:** `lw_saliency` reduced from 0.5 → 0.2. In v31, saliency loss (~3.3 raw) constituted ~33% of the total weighted loss but does not directly improve R1 metrics. Reduction frees gradient budget for span losses.

### 6.5 Cross-Sample NCE Contrastive Alignment

```python
# Per-sample text representation: mean over tokens, L2-normalized
txt_rep = L2_normalize(masked_mean(txt_feat))       # (B, hdim)

# Each matched query competes against all B texts in batch
cross_logits = matched_queries @ txt_rep.T / τ       # (N_matched, B)
loss_nce = cross_entropy(cross_logits, batch_ids)    # positive = own sample's text
```

Temperature `τ=0.15` (Charades), `τ=0.07` (QVH).

### 6.6 Hyperbolic Entailment Loss (3 Components)

```python
ENTAIL_MARGIN = 0.05

# 1. Global positive: text inside video's cone
loss_pos = clamp(angle − aperture + margin, min=0).mean()

# 2. Hard negative: closest wrong text outside cone
hard_neg_idx = argmin(pairwise_dist, exclude_diagonal)
loss_neg = clamp(aperture − neg_angle + margin, min=0).mean()

# 3. Span-level: each matched span must entail text
for matched (b, src_idx):
    sp_angle = oxy_angle(hyp_span[b, src_idx], hyp_txt[b], κ)
    loss_span_ent += clamp(sp_angle − sp_aperture + margin, min=0)

loss_entailment = loss_pop_coef × (loss_pos + loss_neg + loss_span_ent)
```

### 6.7 Hyperbolic Saliency Loss

```python
loss_hyp_sal = clamp(margin + hyp_sal[neg_frame] − hyp_sal[pos_frame], min=0).mean()
```

### 6.8 Hungarian Matching

Bipartite assignment between Q predictions and N_gt targets per sample:

```python
C = cost_class × w_class + cost_span × w_span + cost_giou × w_giou
# cost_class = −sigmoid(pred_logits)       quality score
# cost_span  = L1(pred_cxw, gt_cxw)        center-width distance
# cost_giou  = −GIoU(pred_xx, gt_xx)       temporal IoU

indices = scipy.optimize.linear_sum_assignment(C)  # per batch
```

Matcher costs are dynamically scheduled (see Section 7.3).

**Source:** `Models/matcher.py` — class `HungarianMatcher`

---

## 7. Training Strategy

### 7.1 Optimizer & Scheduler

**Source:** `main_vmr.py` — `build_optimizer`, `build_scheduler`

| Setting | Charades-STA v32 | QVHighlights |
|---------|-----------------|--------------|
| Optimizer | AdamW | AdamW |
| Base LR | **8e-5** *(v31: 1e-4)* | 1e-4 |
| Video encoder LR | **4e-5** *(v31: 5e-5)* | 1e-4 |
| Weight decay | **1e-4** *(v31: 5e-6)* | 1e-4 |
| Gradient clip | 0.3 | 0.1 |
| Warmup | **5 epochs** *(v31: 3)* | 0 |
| LR schedule | **Cosine annealing** *(v31: step ×0.5 at ep25)* | step ×0.1 at ep40 |
| Cosine T0 | **30** epochs | — |
| Cosine T_mult | **2** (cycle doubles) | — |
| Cosine η_min | **1% of base** | — |
| Max epochs | 100 | 200 |
| Early stopping | 50 epochs patience | 50 epochs patience |
| Batch size | 32 | 32 |

**Cosine annealing schedule** (v32):
```
Epochs [0, 5):     linear warmup 0 → base_lr
Epochs [5, 35):    cosine cycle 1: base_lr → 0.01 × base_lr
Epochs [35, 95):   cosine cycle 2 (T=60): base_lr → 0.01 × base_lr
Epochs [95, ...):  cosine cycle 3 (T=120): ...
```

Smooth decay avoids the abrupt LR drop that caused instability in v31 at epoch 25.

### 7.2 EMA (Exponential Moving Average) — v32 New

**Source:** `main_vmr.py` — class `ModelEMA`

```python
# After each optimizer step:
ema_param = decay × ema_param + (1 − decay) × model_param   # decay=0.9995

# Evaluation and checkpointing use EMA weights, not training weights
eval_model = ema.module
```

- Created before training loop; updated every step
- All validation runs and checkpoint saves use EMA model
- Reduces epoch-to-epoch fluctuation (v31 showed ±2 R1@0.5 variance between consecutive epochs)

### 7.3 Primary Metric for Model Selection

```python
primary = (R1@0.5 + R1@0.7) / 2
```

Jointly optimizes both IoU thresholds, preventing gaming of a single one.

Two checkpoints saved:
- `best.ckpt`: best primary metric (EMA model weights)
- `best_saliency.ckpt`: best `hl_mAP` (saliency, EMA model weights)

### 7.4 Multi-Phase Loss Schedule (Charades-STA, v32)

Each entry: `(from_epoch, {cfg_key: new_value})` — applied by `apply_loss_schedule()` at phase boundaries.

**Phases spread out vs v31** (was 0/10/20/30) to match the slower cosine LR decay. Boundary refinement delayed to epoch 15.

**Phase 1 (Epochs 0–14): Coarse Localization**

Span/IoU/label losses active. Boundary refinement OFF until decoder converges.

```
span_loss_coef=10.0, giou_loss_coef=4.0
boundary_refine_coef=0.0, boundary_refine_giou_coef=0.0   ← OFF
contrastive_align_loss_coef=0.05, loss_pop_coef=0.1
aux_loss_scale=0.2
set_cost_class=2.0, set_cost_span=10.0, set_cost_giou=2.0
```

**Phase 2 (Epochs 15–29): Boundary Refinement Activation**

Decoder has converged; activate refinement head at moderate weight.

```
boundary_refine_coef=4.0, boundary_refine_giou_coef=4.0   ← ON
aux_loss_scale=0.3, set_cost_class=3.0
```

**Phase 3 (Epochs 30–49): IoU Dominant**

Shift gradient emphasis toward boundary precision; taper alignment losses.

```
span_loss_coef=8.0, giou_loss_coef=5.0
boundary_refine_coef=8.0, boundary_refine_giou_coef=8.0
contrastive_align_loss_coef=0.02, loss_pop_coef=0.05
set_cost_class=4.0, set_cost_span=8.0, set_cost_giou=3.0
```

**Phase 4 (Epochs 50+): Maximum Boundary Precision**

Peak refinement weight under low LR (cosine trough).

```
span_loss_coef=6.0, giou_loss_coef=6.0
boundary_refine_coef=10.0, boundary_refine_giou_coef=10.0
```

### 7.5 Loss Weights (v32 Phase 1)

| Loss Key | Weight | Notes |
|----------|--------|-------|
| `loss_span` | 10.0 | L1 regression |
| `loss_giou` | 4.0 | 0.5×DIoU + 0.5×Alpha-IoU |
| `loss_boundary` | 0.0 | disabled |
| `loss_label` | 4.0 | quality score BCE |
| `loss_saliency` | **0.2** *(v31: 0.5)* | 3-component saliency |
| `loss_entailment` | 1.0 | (coef applied internally via loss_pop_coef=0.1) |
| `loss_hyp_saliency` | 0.5 | hyperbolic saliency ranking |
| `loss_boundary_refined` | **0.0** *(activated at ep15)* | boundary refinement smooth-L1 |
| `loss_giou_refined` | **0.0** *(activated at ep15)* | boundary refinement Alpha-IoU |
| `loss_contrastive_align` | 0.05 | cross-sample NCE |
| Auxiliary layers (5×) | 0.2× above | deep supervision (excludes saliency, boundary refined) |

### 7.6 Auxiliary Loss Scaling

Intermediate decoder layers (0 through 4, with layer 5 being final) receive the same losses as the final layer, scaled by `aux_loss_scale`:

- Phase 1: 0.2× (prevent early overfitting)
- Phase 2: 0.3× (converging, ramp up)

Excluded from aux: `loss_saliency`, `loss_boundary_refined`, `loss_giou_refined`.

### 7.7 Regularization (v32)

| Technique | v31 Charades | v32 Charades | QVHighlights |
|-----------|-------------|-------------|--------------|
| Input dropout | 0.25 | **0.35** | 0.5 |
| Internal dropout | 0.2 | **0.3** | 0.1 |
| Label smoothing | 0.1 | **0.2** | 0.0 |
| Text token dropout | 0.0 | **0.15** | 0.0 |
| Weight decay | 5e-6 | **1e-4** | 1e-4 |
| Feature noise σ | 0.0 | **0.05** | 0.0 |
| EMA decay | — | **0.9995** | — |

---

## 8. Data Processing

**Source:** `Datasets/vmr_data_provider.py` — class `VMRDataset`

### 8.1 Feature Loading

**Video features:** loaded from `.npz` files, multiple sources concatenated:
- Charades: SlowFast (2304) + CLIP (512) + BLIP (768) = 3584-D
- QVHighlights: CLIP (512) only
- Multi-stream alignment: `scipy.ndimage.zoom` (linear interpolation) to match lengths
- Truncate to `max_v_l=75`; L2-normalize if `normalize_v=True`

**Query features:** loaded from `qid<qid>.npz`:
- Charades: CLIP text (512) + BLIP text (768) = 1280-D
- QVHighlights: CLIP text (512) only
- Truncate to `max_q_l=32`; L2-normalize if `normalize_t=True`

### 8.2 Training Data Augmentation (v32)

Three augmentation stages applied in `__getitem__` during training only:

**1. Temporal Crop** (`temporal_crop_ratio=0.2`, 50% chance per sample):

```python
# Remove up to 20% of frames from start/end, keeping GT window intact
cropped_feat, crop_offset, new_len = _temporal_crop(video_feat, gt_windows, ...)
# GT windows rescaled: start -= crop_offset * clip_len, end -= crop_offset * clip_len
```

Forces context-invariant localization.

**2. Feature Masking** (`feat_mask_ratio=0.15`):

```python
# Zero out 15% of randomly chosen clip features
n_mask = max(1, round(n_clips * 0.15))
video_feat[random_idx] = 0.0
```

Prevents reliance on specific frame patterns.

**3. GT Boundary Jitter** (`gt_jitter_frames=2`):

```python
# Randomly shift GT boundaries by ±2 frames
jitter_sec = gt_jitter_frames * clip_len      # ±2 seconds (clip_len=1.0)
start_new = clamp(start + uniform(-jitter_sec, jitter_sec), 0, duration)
end_new   = clamp(end   + uniform(-jitter_sec, jitter_sec), start_new + clip_len, duration)
```

Smooths the supervision signal, reducing overconfident prediction of exact boundaries.

### 8.3 Label Generation

**Span labels:** temporal windows `[start_sec, end_sec]` → normalized `(center, width)` in [0, 1].

**Saliency labels:**
- QVHighlights: multiple annotator scores → aggregate per clip; hard pos/neg selection based on scores
- Charades-STA: binary from GT window (1.0 inside, 0.0 outside); random pos/neg clip sampling

### 8.4 Collation

- Video features: padded to fixed `max_v_l=75` (required by HLFormerBlock's `Linear(H, frame_len)` layer)
- Query features: padded to batch maximum
- Span labels: kept as list of dicts per sample (variable GT count)

### 8.5 Post-Processing

```
pred_logits → sigmoid → foreground probability
pred_spans_refined → (center, width) → (start, end) in seconds
→ Sort by probability descending → Top-K (K=5 for Charades v32)
→ Temporal NMS (IoU threshold=0.5 v32) → final predictions
```

**Source:** `Validations/vmr_validations.py` — `post_process_predictions`, `temporal_nms`

---

## 9. Summary: Key Innovations

| Category | Component | Description |
|----------|-----------|-------------|
| **Encoder** | HLFormerBlock | 8-branch (4 Euclidean + 4 Hyperbolic) with per-position learned fusion |
| | Gaussian windowed attention | Multi-scale local attention (wid=2,4,8) + global (no window) |
| | Lorentzian attention | Attention on hyperboloid with Fréchet mean aggregation |
| | Per-position fusion | CrossAttention + MLP selects optimal branch mix per frame |
| **Decoder** | Content-Adaptive Query (CAQ) | Text-gated, video-attended query initialization |
| | Adaptive reference points | CAQ-derived initial reference offset |
| | 6-layer iterative refinement | Sine-conditioned cross-attention with per-layer span prediction |
| **Refinement** | BoundaryRefinementHead | Gaussian pooling around predicted boundaries + clamped MLP delta |
| | Delayed activation (v32) | Activated at epoch 15; prevents interference with decoder convergence |
| **Geometry** | Lorentz hyperboloid model | Numerically stable hyperbolic space for hierarchical features |
| | Alpha scaling | Log-parameterized, ≤0-constrained feature scaling before projection |
| | Hyperbolic entailment | Cone geometry: text inside video's (and span's) entailment cone |
| | Hard negative mining | Closest-in-batch negatives (hyperbolic distance) |
| **Losses** | DIoU + Alpha-IoU | Combined IoU loss targeting center alignment + medium-IoU gradient amplification |
| | Quality score (soft IoU target) | IoU-proportional classification replacing binary fg/bg |
| | Cross-sample NCE | Contrastive alignment using full batch as negatives |
| | Hyperbolic saliency | Lorentzian inner product as auxiliary saliency channel |
| | Fixed rank contrastive | Divide by actual active thresholds (not hardcoded) |
| **Training** | Cosine annealing LR (v32) | Smooth decay with warm restarts; avoids abrupt step-drop instability |
| | EMA averaging (v32) | 0.9995 decay; eval/checkpoint on EMA weights; reduces val fluctuation |
| | Multi-phase loss schedule | 4-phase curriculum: coarse → boundary-on → IoU-dominant → fine |
| | Dynamic matcher costs | `set_cost_class` ramped 2.0 → 4.0 as quality head converges |
| | Ramped aux_loss_scale | 0.2 → 0.3 to prevent intermediate layer overfitting |
| **Data** | Multi-stream fusion | Independent projection + learned fusion of heterogeneous features |
| | Simplified negative saliency | Shifted dot product instead of full encoder re-run (~40% speedup) |
| | Feature noise (v32) | Gaussian noise σ=0.05 on projected features (training only) |
| | Temporal crop augmentation (v32) | Crop up to 20% of video; GT windows rescaled accordingly |
| | Feature masking (v32) | Zero 15% of clips randomly; prevents frame-pattern memorization |
| | GT boundary jitter (v32) | ±2 frame perturbation; smooths supervision signal |

---

## 10. Hyperparameter Reference

### 10.1 Model Architecture (Charades-STA v32)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hidden_size` | 384 | embedding dimension |
| `n_heads` | 8 | 384/8 = 48 per head |
| `num_queries` | **5** *(v31: 10)* | detection query slots; 1 GT/sample typical |
| `dec_layers` | 6 | decoder refinement iterations |
| `t2v_layers` | 6 | T2V cross-attention layers |
| `attention_num` | 8 | 4 Euclidean + 4 Hyperbolic branches |
| `sft_factor` | 0.1 | branch fusion softmax temperature |
| `lorentz_dim` | 191 | H//2 − 1 for hyperbolic features |
| `input_drop` | **0.35** *(v31: 0.25)* | input projection dropout |
| `drop` | **0.3** *(v31: 0.2)* | internal dropout |
| `txt_drop_ratio` | **0.15** *(v31: 0.0)* | text token zeroing probability |
| `pos_enc_type` | trainable | learned positional encoding |
| `max_v_l` | 75 | max video length (frames) |
| `max_q_l` | 32 | max query length (tokens) |
| `clip_len` | 1.0 s | seconds per clip (Charades) |

### 10.2 Loss Weights (v32 Phase 1, Epochs 0–14)

| Parameter | v32 Value | v31 Value | Notes |
|-----------|-----------|-----------|-------|
| `span_loss_coef` | 10.0 | 10.0 | L1 span regression |
| `giou_loss_coef` | 4.0 | 4.0 | DIoU + Alpha-IoU combined |
| `label_loss_coef` | 4.0 | 4.0 | quality score BCE |
| `lw_saliency` | **0.2** | 0.5 | 3-component saliency |
| `boundary_refine_coef` | **0.0** | 4.0 | disabled until ep15 |
| `boundary_refine_giou_coef` | **0.0** | 4.0 | disabled until ep15 |
| `contrastive_align_loss_coef` | 0.05 | 0.05 | cross-sample NCE |
| `loss_pop_coef` | 0.1 | 0.1 | entailment coefficient |
| `hyp_saliency_coef` | 0.5 | 0.5 | hyperbolic saliency |
| `alpha_iou_alpha` | 2.5 | 2.5 | Alpha-IoU exponent |
| `label_smoothing` | **0.2** | 0.1 | quality score smoothing |
| `ENTAIL_MARGIN` | 0.05 | 0.05 | entailment hinge margin |
| `temperature` | 0.15 | 0.15 | contrastive NCE temperature |
| `saliency_margin` | 1.0 | 1.0 | saliency ranking margin |
| `aux_loss_scale` | 0.2 | 0.2 | intermediate layer loss scale |

### 10.3 Boundary Refinement Head (v32)

| Parameter | v32 Value | v31 Value | Notes |
|-----------|-----------|-----------|-------|
| `boundary_refine_window` | **15** | 10 | Gaussian σ init = 15/(2×75) ≈ 7.5 frames |
| `boundary_refine_max_delta` | **0.25** | 0.15 | ±19 frames correction range |
| `boundary_refine_learnable_sigma` | True | True | σ learned per boundary |
| Activation epoch | **15** | 0 | delayed to let decoder converge |

### 10.4 Hyperbolic Geometry

| Parameter | Charades | QVHighlights | Notes |
|-----------|----------|--------------|-------|
| `lorentz_dim` | 191 (H//2−1) | 127 | hyperbolic feature dim |
| `curv_init` | 1.0 | 1.0 | log-space: κ=exp(1.0)≈2.718 |
| `learn_curv` | False | True | fixed κ more stable for smaller dataset |
| `curv_clamp` | [κ/10, κ×10] | same | prevent curvature divergence |
| `video_alpha` init | log(H^−0.5) | same | ≈ −2.95 for H=384 |

### 10.5 Optimizer (Charades-STA v32)

| Parameter | v32 Value | v31 Value |
|-----------|-----------|-----------|
| `lr` | **8e-5** | 1e-4 |
| `lr_vid_enc` | **4e-5** | 5e-5 |
| `wd` | **1e-4** | 5e-6 |
| `warmup_epochs` | **5** | 3 |
| `cosine_T0` | **30** | — |
| `cosine_Tmult` | **2** | — |
| `cosine_eta_min_ratio` | **0.01** | — |
| `lr_drop` | 25 (unused) | 25 |
| `lr_gamma` | 0.5 (unused) | 0.5 |
| `grad_clip` | 0.3 | 0.3 |
| `use_ema` | **True** | — |
| `ema_decay` | **0.9995** | — |

### 10.6 Post-Processing (Charades-STA v32)

| Parameter | v32 Value | v31 Value | Notes |
|-----------|-----------|-----------|-------|
| `top_k` | **5** | 7 | matches num_queries=5 |
| `nms_thresh` | **0.5** | 0.3 | lenient; R1 cares only about top-1 |
| `iou_thresholds` | [0.5, 0.7] | [0.5, 0.7] | unchanged |

### 10.7 Hungarian Matcher (v32 Phase 1 → Phase 4)

| Parameter | Phase 1 (ep 0–14) | Phase 2 (ep 15–29) | Phase 3 (ep 30–49) | Phase 4 (ep 50+) |
|-----------|-------------------|--------------------|--------------------|--------------------|
| `set_cost_class` | 2.0 | 3.0 | 4.0 | 4.0 |
| `set_cost_span` | 10.0 | 10.0 | 8.0 | 8.0 |
| `set_cost_giou` | 2.0 | 2.0 | 3.0 | 3.0 |

### 10.8 Data Augmentation (v32, Charades-STA training only)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temporal_crop_ratio` | 0.2 | crop up to 20% of frames (50% chance) |
| `feat_mask_ratio` | 0.15 | zero 15% of clips randomly |
| `gt_jitter_frames` | 2 | ±2 frame boundary perturbation |
| `feat_noise_std` | 0.05 | Gaussian noise σ on projected features |
