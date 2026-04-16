# HLFormer-VMR Architecture

Video Moment Retrieval model that combines HLFormer's hyperbolic multi-scale encoder with QD-DETR's detection decoder and Early Query Fusion.

**Sources of origin per component:**
- HLFormer backbone (Euclidean + Hyperbolic attention) → from `ICCV25-HLFormer`
- T2V Encoder, Decoder, Prediction heads, Saliency loss → from `QD-DETR`
- Hyperbolic entailment head and loss → from `ICCV25-HLFormer`

---

## Overview

```
src_vid  (B, L_v, D_v)  ──┐
src_txt  (B, L_t, D_t)  ──┤
src_vid_mask (B, L_v)   ──┤
src_txt_mask (B, L_t)   ──┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 1 · Dual Feature Projection                                │
├───────────────────────────────────────────────────────────────────┤
│  Stage 2 · Text Self-Attention  (Query Encoder)                   │
├───────────────────────────────────────────────────────────────────┤
│  Stage 3 · Early Query Fusion  (T2V Encoder) ◄── NEW              │
├───────────────────────────────────────────────────────────────────┤
│  Stage 4 · Video Encoder  (HLFormerBlock — CORE INNOVATION)       │
├───────────────────────────────────────────────────────────────────┤
│  Stage 5 · Global Cross-Modal Token  (for saliency)               │
├───────────────────────────────────────────────────────────────────┤
│  Stage 6 · Cross-Modal Decoder  (learned query slots)             │
├───────────────────────────────────────────────────────────────────┤
│  Stage 7 · Prediction Heads  (span + class)                       │
├───────────────────────────────────────────────────────────────────┤
│  Stage 8 · Saliency Scoring  (positive + negative pairs)          │
├───────────────────────────────────────────────────────────────────┤
│  Stage 9 · Hyperbolic Entailment Head  (optional)                 │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
pred_logits  (B, Q, 2)
pred_spans   (B, Q, 2)   normalized (center, width) in [0, 1]
saliency_scores     (B, L_v)
saliency_scores_neg (B, L_v)
hyp_vid_feat / hyp_txt_feat  (B, H) | None
aux_outputs  [list of intermediate decoder dicts]
```

---

## Stage 1 — Dual Feature Projection

Both video and text features are independently projected to the shared hidden dimension `H=256`, then position-encoded.

```
src_vid (B, L_v, D_v) ──► LinearLayer(D_v → H, LN+ReLU+Drop) ──► vid_pos_embed ──► (B, L_v, H)
src_txt (B, L_t, D_t) ──► LinearLayer(D_t → H, LN+ReLU+Drop) ──► txt_pos_embed ──► (B, L_t, H)
```

| Symbol | Value (QVHighlights) | Value (Charades) |
|--------|---------------------|------------------|
| `D_v`  | 512 (CLIP)          | 4096 (VGG)       |
| `D_t`  | 512 (CLIP)          | 512 (CLIP)       |
| `H`    | 256                 | 256              |
| `L_v`  | ≤ 75                | ≤ 64             |
| `L_t`  | ≤ 32                | ≤ 32             |

`LinearLayer` = `Linear → LayerNorm → ReLU → Dropout(input_drop=0.5)`
`TrainablePositionalEncoding` = learnable position embeddings + dropout.

---

## Stage 2 — Text Self-Attention (Query Encoder)

Standard Euclidean self-attention over the text sequence. Unchanged from HLFormer.

```
txt_feat (B, L_t, H) ──► EuclideanAttentionBlock ──► txt_feat (B, L_t, H)
                          mask: src_txt_mask (B, 1, L_t)  [1=valid]
```

`EuclideanAttentionBlock` = multi-head self-attention + FFN + LayerNorm.
Config: `n_heads=8`, `drop=0.1`.

---

## Stage 3 — Early Query Fusion (T2V Encoder)

Injects text context into every video clip **before** hierarchical encoding.
Mirrors QD-DETR's `T2V_TransformerEncoderLayer` stack.

**Roles:**
- Video tokens → **Query**
- Text tokens → **Key** and **Value**

```
txt_feat ────────────────────────────────────────────┐
                                                     │  K, V
vid_feat ──► T2VEncoderLayer 0                       │
             ├─ MultiheadAttention(Q=vid, K=txt, V=txt) ◄─┘
             │   key_padding_mask: (B, L_t) True=padded text
             │   attn_mask:        (B·H, L_v, L_t)  vid_pad ∧ txt_pad
             ├─ residual + LayerNorm
             ├─ FFN: Linear(H→4H) → PReLU → Linear(4H→H)
             └─ residual + LayerNorm
                       │
             T2VEncoderLayer 1  (same structure)
                       │
                       ▼
             vid_feat* (B, L_v, H)   [each clip is now text-aware]
```

**Padding-aware 2D attention mask:**

```python
vid_pad   = (vid_mask == 0).unsqueeze(2)   # (B, L_v, 1) True=padded clip
txt_pad   = (txt_mask == 0).unsqueeze(1)   # (B, 1, L_t) True=padded token
attn_mask = vid_pad & txt_pad              # (B, L_v, L_t) outer product
# expanded → (B·n_heads, L_v, L_t)
```

Blocks padded-clip × padded-token pairs. Together with `key_padding_mask`
(which already blocks all-clip → padded-token), this prevents `softmax(-inf)`
→ NaN when an entire text sequence is padding.

Config: `t2v_layers=2`, `n_heads=8`, `FFN dim=4H=1024`, `dropout=0.1`.

---

## Stage 4 — Video Encoder (HLFormerBlock)

Core innovation from HLFormer. Processes the **text-conditioned** video features
through 8 parallel attention branches — 4 Euclidean and 4 Hyperbolic — with
per-position learned fusion weights.

```
vid_feat* (B, L_v, H)
    │
    ├──► EuclideanGaussianAttention branch 0  (Gaussian window scale 0) ──► h_e0
    ├──► EuclideanGaussianAttention branch 1  (Gaussian window scale 1) ──► h_e1
    ├──► EuclideanGaussianAttention branch 2  (Gaussian window scale 2) ──► h_e2
    ├──► EuclideanGaussianAttention branch 3  (Gaussian window scale 3) ──► h_e3
    │
    ├──► LorentzSelfAttention branch 0  (lorentz_dim=127) ──► h_l0
    ├──► LorentzSelfAttention branch 1                    ──► h_l1
    ├──► LorentzSelfAttention branch 2                    ──► h_l2
    └──► LorentzSelfAttention branch 3                    ──► h_l3
              │
              ▼
    concat [h_e0, h_e1, h_e2, h_e3, h_l0, h_l1, h_l2, h_l3]
              │
    Linear(8H → L_v)  × sft_factor=0.09  →  softmax over 8 branches  (B, L_v, 8)
              │
    per-position weighted sum  →  FFN  →  vid_feat (B, L_v, H)
              │
    × src_vid_mask  (zero-out padded positions)
```

Config: `attention_num=8` (4+4), `lorentz_dim=127`, `sft_factor=0.09`,
`frame_len=max_v_l` (fixed-length required for branch-weight linear).

---

## Stage 5 — Global Cross-Modal Token

A single learnable vector `global_token ∈ ℝᴴ` cross-attends to the full
video+text context to produce a compact cross-modal summary used in saliency.

```
global_token (H,)  ──► expand ──► (B, 1, H)   [learnable parameter]
                            │ Q
CrossAttention ─────────────┘
    K, V ◄──── cat(vid_feat, txt_feat)  (B, L_v+L_t, H)
    mask: cat(vid_mask, txt_mask).unsqueeze(1)  (B, 1, L_v+L_t)
                            │
                            ▼
                    global_rep (B, H)
```

---

## Stage 6 — Cross-Modal Decoder

`num_queries=10` learned detection slots (query embeddings) attend to the
concatenated video+text memory via a stack of `TransformerDecoderLayer` blocks.

```
query_embed  Embedding(10, H)  ──► expand ──► tgt (B, 10, H)

memory      = cat(vid_feat, txt_feat)  (B, L_v+L_t, H)
memory_mask = cat(vid_mask, txt_mask)  (B, L_v+L_t)

┌── TransformerDecoderLayer 0 ─────────────────────────────────┐
│   self-attn(tgt, tgt)  [query-slot self-attention]           │
│   cross-attn(tgt, memory)  [slots attend to vid+txt]         │
│   FFN  (dim_feedforward=4H)                                  │
│   [save output if aux_loss=True]                             │
└──────────────────────────────────────────────────────────────┘
            │
┌── TransformerDecoderLayer 1 ─────────────────────────────────┐
│   (same structure)                                           │
└──────────────────────────────────────────────────────────────┘
            │
        LayerNorm
            │
        dec_out (B, 10, H)
```

Config: `dec_layers=2`, `n_heads=8`, `FFN dim=4H`, `drop=0.1`.
PyTorch `key_padding_mask` convention: `True = ignore` (inverted from HLFormer mask).

---

## Stage 7 — Prediction Heads

```
dec_out (B, Q, H)
    ├──► class_head  Linear(H → 2)                 ──► pred_logits (B, Q, 2)
    │                                                   [fg=0 / bg=1 logits]
    └──► span_head   MLP(H → H → H → 2) + sigmoid  ──► pred_spans  (B, Q, 2)
                                                        [(center, width) ∈ [0,1]]
```

Span format: normalized `(center, width)` — converted to `(start_sec, end_sec)`
in seconds during post-processing via `span_cxw_to_xx(spans) × duration`.

---

## Stage 8 — Saliency Scoring

Per-clip relevance scores computed for both a **positive pair** (correct query)
and a **negative pair** (cyclically shifted query within the batch).

```
Positive pair:
  s_pos = Σ_h [ saliency_proj1(vid_feat)_h · saliency_proj2(global_rep)_h ]
        / sqrt(H)                                              → (B, L_v)

Negative pair  (roll batch by 1: sample i pairs with query i+1 mod B):
  txt_neg      = roll(src_txt, shift=1, dim=0)
  global_neg   = CrossAttention(global_token, cat(vid, txt_neg))
  s_neg = Σ_h [ saliency_proj1(vid_feat)_h · saliency_proj2(global_neg)_h ]
        / sqrt(H)                                              → (B, L_v)
```

Both scores are used in the saliency loss: `s_pos[valid] > s_neg[valid] + margin`.

---

## Stage 9 — Hyperbolic Entailment Head (optional)

When `use_hyperbolic=True`, video and text are mapped to the Lorentz hyperboloid
for entailment regularization (text cone ⊂ video cone).

```
Attention-pooled video global vector:
  v_w         = softmax(video_pool(vid_feat))            (B, L_v, 1)
  v_g         = (vid_feat × v_w).sum(dim=1)              (B, H)
              × exp(video_alpha)                          (learnable scaling)
  hyp_vid     = exp_map0(v_g, κ)                         on Lorentz hyperboloid

Attention-pooled text global vector (text-mask aware):
  t_score     = query_pool(txt_feat)                     (B, L_t, 1)
              masked: padded positions → -1e4
  t_w         = softmax(t_score)
  t_g         = (txt_feat × t_w).sum(dim=1)              (B, H)
              × exp(textual_alpha)
  hyp_txt     = exp_map0(t_g, κ)                         on Lorentz hyperboloid

κ = exp(curv)   learnable curvature, clamped to [curv_init/10, curv_init×10]
```

The entailment loss (`loss_pop`) penalizes `angle(hyp_txt) > aperture(hyp_vid)`
on the hyperboloid — enforcing that text is "entailed by" video in hyperbolic space.

---

## Output Dictionary

| Key                  | Shape        | Description                                       |
|----------------------|--------------|---------------------------------------------------|
| `pred_logits`        | `(B, Q, 2)`  | fg / bg classification logits per query slot      |
| `pred_spans`         | `(B, Q, 2)`  | normalized (center, width) spans, sigmoid output  |
| `saliency_scores`    | `(B, L_v)`   | per-clip relevance (positive query pair)          |
| `saliency_scores_neg`| `(B, L_v)`   | per-clip relevance (negative query pair)          |
| `video_mask`         | `(B, L_v)`   | valid clip mask (passthrough from input)          |
| `hyp_vid_feat`       | `(B, H)`     | video point on Lorentz hyperboloid (or `None`)    |
| `hyp_txt_feat`       | `(B, H)`     | text point on Lorentz hyperboloid (or `None`)     |
| `_curv`              | scalar       | current curvature κ (or `None`)                   |
| `aux_outputs`        | `list[dict]` | intermediate decoder `{pred_logits, pred_spans}`  |

---

## Loss Functions

Defined in `VMR/Losses/vmr_loss.py` (`VMRSetCriterion`).

| Loss | Weight | Description |
|------|--------|-------------|
| `loss_spans` | `span_loss_coef=10` | L1 regression on Hungarian-matched spans |
| `loss_giou` | `giou_loss_coef=1` | Generalized temporal IoU on matched spans |
| `loss_labels` | `label_loss_coef=4` | fg/bg cross-entropy (`eos_coef=0.1` for bg) |
| `loss_saliency` | `lw_saliency=1` | neg-pair + ranking contrastive + margin loss |
| `loss_entailment` | `loss_pop_coef=1e-3` | hyperbolic cone entailment (optional) |
| aux losses | same weights | intermediate decoder layer losses (if `aux_loss=True`) |

Hungarian matching costs: `set_cost_class=4`, `set_cost_span=10`, `set_cost_giou=1`.

---

## Hyperparameters (QVHighlights defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_size` | 256 | shared feature dimension H |
| `n_heads` | 8 | attention heads (all modules) |
| `input_drop` | 0.5 | projection dropout |
| `drop` | 0.1 | internal dropout |
| `t2v_layers` | 2 | T2V encoder depth (Stage 3) |
| `dec_layers` | 2 | decoder depth (Stage 6) |
| `num_queries` | 10 | detection query slots |
| `max_v_l` | 75 | max video clips (QVH) / 64 (Charades) |
| `max_q_l` | 32 | max query tokens |
| `attention_num` | 8 | HLFormerBlock branches (4E + 4H) |
| `lorentz_dim` | 127 | hyperbolic feature dimension |
| `sft_factor` | 0.09 | branch-weight scaling |
| `curv_init` | 1.0 | initial hyperbolic curvature κ |
| `learn_curv` | True | learnable curvature |
| `use_hyperbolic` | True | enable entailment head + loss |
| `aux_loss` | True | intermediate decoder losses |
| `lr` | 1e-4 | base learning rate |
| `lr_vid_enc` | 1e-4 | video encoder LR |
| `wd` | 1e-4 | weight decay |
| `grad_clip` | 0.1 | gradient norm clipping |
| `n_epoch` | 200 | max training epochs |
| `max_es_cnt` | 50 | early stopping patience |
| `batchsize` | 32 | training batch size |

---

## File Map

```
src/VMR/
├── ARCHITECTURE.md           ← this file
├── main_vmr.py               ← training entry point
├── Configs/
│   ├── qvhighlights.py       ← QVHighlights config (base)
│   └── charades.py           ← Charades-STA config (inherits qvhighlights)
├── Datasets/
│   └── vmr_data_provider.py  ← VMRDataset, collate, prepare_batch_inputs
├── Models/
│   ├── vmr_model.py          ← HLFormer_VMR, T2VEncoderLayer, MLP
│   ├── matcher.py            ← HungarianMatcher (bipartite assignment)
│   └── span_utils.py         ← span_cxw_to_xx, temporal_iou, GIoU
├── Losses/
│   └── vmr_loss.py           ← VMRSetCriterion (all losses)
└── Validations/
    └── vmr_validations.py    ← evaluate_vmr, R1@IoU, mAP, hl_mAP, HIT@1
```

---

## Mask Conventions

| Location | Convention | Shape |
|----------|-----------|-------|
| Input masks | `1=valid, 0=padded` (float) | `(B, L)` |
| HLFormer blocks (`EuclideanAttentionBlock`, `CrossAttention`) | `1=valid, 0=padded` | `(B, 1, L)` — must unsqueeze |
| `nn.MultiheadAttention` `key_padding_mask` | `True=padded` (bool) | `(B, L)` |
| `nn.MultiheadAttention` `attn_mask` | `True=ignore` (bool) | `(B·H, L_q, L_k)` |
| PyTorch `TransformerDecoderLayer` `key_padding_mask` | `True=padded` (bool) | `(B, L)` |

The inversion `(mask == 0)` is applied at the boundary between HLFormer-style
and PyTorch-style modules.
