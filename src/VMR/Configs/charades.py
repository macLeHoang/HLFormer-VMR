"""
Configuration for HLFormer-VMR on Charades-STA dataset.

Charades-STA features:
  Video: SlowFast (2304) + CLIP (512) + BLIP (768)  -> <vid>.npz  (key: "features")
  Text:  CLIP (512) + BLIP (768)                    -> <qid>.npz  (key: "last_hidden_state")
"""

import os
import yaml
# Import the raw dict without calling get_cfg_defaults() to avoid side effects
from VMR.Configs import qvhighlights as _qvh

cfg = dict(_qvh.cfg)   # inherit defaults, then override

cfg["seed"]          = 2026

cfg["model_name"]    = "HLFormer_VMR_SFCB_v35"   # v34: boundary_refine=4.0 from ep0, saliency=0.05, max_v_l=128, cosine_T0=50, warmup=10, lr=2e-4, alpha_iou=3.0, ema_decay=0.9995
cfg["dataset_name"]  = "charades_sta"
cfg["dset_name"]     = "charades_sta"

cfg["data_root"]   = ""
cfg["v_feat_dirs"] = [
    "/content/charades/slowfast_features",
    "/content/charades/clip_features",
    "/content/charades/blip_video_features",
]
cfg["q_feat_dir"]  = [
    "/content/charades/clip_text_features",
    "/content/charades/blip_text_features",
]

cfg["train_path"]  = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/train.jsonl"
cfg["val_path"]    = "/content/drive/MyDrive/Master/Thesis/QD-DETR/data/charades-sta/test.jsonl"
cfg["test_path"]   = None

cfg["model_root"]  = os.path.join(cfg["root"], cfg["dataset_name"], cfg["model_name"])

# Per-stream dims drive MultiStreamVidProjection; v_feat_dim/t_feat_dim are
# kept as totals for logging and any legacy code that reads them.
cfg["v_feat_dims"] = [2304, 512, 768]            # SlowFast | BLIP | CLIP  (order must match v_feat_dirs)
cfg["v_feat_dim"]  = sum(cfg["v_feat_dims"])     # 3584
cfg["t_feat_dims"] = [512, 768]                  # BLIP text | CLIP text   (order must match q_feat_dir)
cfg["t_feat_dim"]  = sum(cfg["t_feat_dims"])     # 1280

cfg["max_v_l"]     = 75
cfg["clip_len"]    = 1.0

# ---- Model architecture --------------------------------------------------
cfg["hidden_size"]    = 384   # v30: restored to 384; v28=320 hit capacity ceiling
cfg["n_heads"]        = 8     # 384/8=48 per head
cfg["num_queries"]    = 5     # v34: restored to 10; more candidates give Hungarian matcher better options
cfg["dec_layers"]     = 4     # v33: 6→4; fewer aux layers reduces gradient interference on 1-GT dataset
cfg["input_drop"]     = 0.25   # v33: 0.35→0.2; heavy dropout was preventing fine boundary learning
cfg["drop"]           = 0.15  # v33: 0.3→0.15; reduce regularization — plateau is generalization, not overfit
cfg["txt_drop_ratio"] = 0.1   # v33: 0.15→0.1; lighter text dropout
cfg["t2v_layers"]     = 4    # v17→v18: deeper cross-modal fusion for richer global_rep
cfg["attention_num"]  = 8
cfg["pos_enc_type"]   = "trainable"
cfg["sft_factor"]     = 0.1   # v11=0.06; stronger query-conditioned feature shift

# v11=False; enabling text-in-memory makes encoder query-conditioned (not just decoder)
cfg["use_txt_in_memory"]     = True
cfg["use_global_in_encoder"] = True

# ---- Hyperbolic entailment -----------------------------------------------
# v14 enhancements:
#   - alpha init fixed to log(H^-0.5) in vmr_model.py (better hyperboloid scaling)
#   - span-level entailment: matched query spans must entail text (→ R1@0.7 signal)
#   - hard negative mining: replace torch.roll with closest-in-batch negatives
#   - hyp_saliency_coef: new loss channel using Lorentzian frame-text similarity
cfg["use_hyperbolic"]     = True
cfg["lorentz_dim"]        = cfg["hidden_size"] // 2 - 1   # 191  (384//2-1; v11=159)
cfg["loss_pop_coef"]      = 0.1    # v31: was 0.005; 20x increase so entailment actually contributes gradient
cfg["curv_init"]          = 1.0   # log-space init; actual curv=exp(1.0)≈2.718 (stable)
cfg["learn_curv"]         = True   # v33: re-enable; fixed curv limits expressiveness; clamp in model prevents collapse
cfg["hyp_saliency_coef"]  = 0.1   # v34: 0.5→0.1; saliency doesn't help R1, free budget for span losses

# ---- Contrastive alignment -----------------------------------------------
# temperature=0.07: MoCo-style sharp negatives; v11=0.3 was too smooth → loss never converged
# contrastive_align_loss_coef=0.01: reduce contrastive dominance over span/boundary losses
cfg["contrastive_hdim"]            = 64
cfg["contrastive_align_loss_coef"] = 0.15  # v33: 0.05→0.15; cross-modal alignment is critical for boundary quality
# temperature for cross-sample NCE (v26 fix):
# The previous 0.1 was calibrated for the broken fg-only logsumexp (which was constant anyway).
# With real cross-sample NCE (B=32 negatives), 0.1 is too sharp — loss collapses to 0.26 by
# epoch 13, meaning the model memorises batch-level text identity rather than learning
# span-text alignment.  0.3 gives softer gradients; loss should stabilise around 2.0–2.5
# (log(32)≈3.47 at random, ~1.5–2.0 at good discrimination without memorisation).
cfg["temperature"]                 = 0.15  # v31: was 0.3; sharper contrastive discrimination

# ---- Saliency ------------------------------------------------------------
cfg["lw_saliency"]     = 0.2   # v34: 0.2→0.05; saliency doesn't contribute to R1, redirect budget to span losses.
cfg["saliency_margin"] = 1.0

# ---- Loss weights --------------------------------------------------------
cfg["span_loss_coef"]     = 10.0  # v17→v18: boost L1 span signal (dominant early training)
cfg["giou_loss_coef"]     = 6.0   # v33: 4.0→6.0; boundary precision is the primary bottleneck for R1@0.7
cfg["boundary_loss_coef"] = 0.0   # disabled: smooth-L1 on (start,end) is redundant with
                                   # DIoU which supervises the same coordinates; the old
                                   # value of 16.0 was swamping the DIoU gradient 16:1.
                                   # Boundary supervision is kept in the refinement head.
cfg["label_loss_coef"]    = 2.0   # v33: 4.0→2.0; quality head converges fast, reduce dominance over span losses
cfg["label_smoothing"]    = 0.2  # v32: 0.1→0.2; softer targets reduce overconfident predictions

# ---- Boundary refinement losses (BoundaryRefinementHead, v15) -----------
# Applied to pred_spans_refined (final layer only, same Hungarian indices).
# v31: enabled from epoch 0 — the BoundaryRefineHead is zero-initialized so it
# starts as identity and gradually activates.  With giou_coef=4.0 the model gets
# direct boundary supervision from the beginning.
cfg["boundary_refine_coef"]           = 4.0   # v34: 1.0→4.0; start at high weight from ep0 — zero-init
                                               # guarantees identity at start, so high weight is safe.
                                               # Evaluation uses pred_spans_refined, so this loss must be strong.
cfg["boundary_refine_giou_coef"]      = 4.0   # v34: same rationale as boundary_refine_coef
cfg["boundary_refine_window"]         = 12     # v34: at max_v_l=77, sigma=16/154≈0.10 (10% of video); tight enough for precise boundary pooling
cfg["boundary_refine_learnable_sigma"] = True
cfg["boundary_refine_max_delta"]      = 0.25  # v34: ±32 frames at max_v_l=128; generous cap allows
                                               # correcting large boundary errors common in Charades
                                               # (avg moment 8s, IoU=0.5→0.7 needs ~4-frame precision)
cfg["alpha_iou_alpha"]                = 3.0   # v34: 2.5→3.0; stronger gradient amplification in 0.5-0.7 IoU zone

# aux_loss_scale starts low: early decoder layers produce near-zero IoU targets,
# creating a strong "everything is background" pull if aux losses are weighted heavily.
# Ramped via loss_schedule once coarse localization stabilises.
cfg["aux_loss_scale"]     = 0.2   # v31: was 0.1; stronger deep supervision from start

# ---- Hungarian matcher ---------------------------------------------------
# v31: set_cost_class and set_cost_giou raised from 1.0 to 2.0 from start
# to stabilize matching early (quality predictions are good enough by ep3-5).
cfg["set_cost_class"]  = 2.0   # v31: was 1.0; stabilize query-target assignment
cfg["set_cost_span"]   = 10.0  # v17→v18: match span_loss_coef ratio
cfg["set_cost_giou"]   = 2.0   # v31: was 1.0; IoU-aware matching from start

# ---- Post-processing -----------------------------------------------------
cfg["top_k"]      = 5   # v34: must match num_queries=10; keep all predictions as NMS candidates
cfg["nms_thresh"] = 0.5  # v32: 0.3→0.5; more lenient NMS — R1 only cares about the top-1 prediction,
                          # tight NMS was removing good predictions that overlapped with slightly worse ones

# ---- Optimizer -----------------------------------------------------------
cfg["lr"]            = 1.5e-4  # v34: 1e-4→2e-4; stronger gradient signal, more budget before cosine trough
cfg["lr_vid_enc"]    = 0.75e-4   # v34: 5e-5→1e-4; proportional to lr increase
cfg["lr_drop"]       = 25     # kept for backward compat; unused when cosine_T0 > 0
cfg["lr_gamma"]      = 0.5    # kept for backward compat
cfg["warmup_epochs"] = 5     # v34: 5→10; longer warmup avoids early loss spikes with higher lr
cfg["cosine_T0"]     = 30     # v34: 30→50; longer first cosine cycle gives more budget before LR trough
cfg["cosine_Tmult"]  = 2      # v32: NEW; second cycle = 60 epochs (total ~90 before 2nd restart)
cfg["cosine_eta_min_ratio"] = 0.01  # v32: NEW; min LR = 1% of base at cycle trough
cfg["wd"]            = 5e-5   # v32: 5e-6→1e-4; 20x increase to regularize weights (5e-6 is effectively zero)
cfg["grad_clip"]     = 0.3

# ---- EMA ----------------------------------------------------------------
cfg["use_ema"]       = True   # v32: NEW; exponential moving average smooths val fluctuations (+1-2 R1)
cfg["ema_decay"]     = 0.999  # v34: 0.999→0.9995; EMA tracks current model more closely during active learning
cfg["n_epoch"]       = 100
cfg["max_es_cnt"]    = 50     # v17→v18: give more room after LR drop
cfg["batchsize"]     = 32

# ---- DataLoader ----------------------------------------------------------
cfg["num_workers"] = 2

# ---- Data augmentation (training only) -----------------------------------
cfg["temporal_crop_ratio"] = 0.2   # v32: randomly crop 20% of video, forces context-invariant localization
cfg["feat_mask_ratio"]     = 0.15  # v32: randomly mask 15% of clips, prevents reliance on specific frames
cfg["gt_jitter_frames"]    = 2     # v32: jitter GT boundaries ±2 frames, smooths supervision signal
cfg["feat_noise_std"]      = 0.0  # v32: Gaussian noise σ added to projected features during training

# ---- Loss schedule -------------------------------------------------------
# Each entry: (from_epoch, {cfg_key: new_value, ...})
# apply_loss_schedule() in main_vmr.py reads this and patches criterion.weight_dict
# and matcher costs at phase boundaries.
#
# v34 schedule: boundary refinement starts at 4.0 from ep0 (zero-init guarantees safety),
# saliency/hyp_saliency deprioritized globally, alpha_iou weighted 70%.
#
# Phase 1  (ep  0-19): Coarse localization — boundary refine at 4.0 (active from start),
#                       zero-init head gradually activates, contrastive alignment strong.
# Phase 2  (ep 20-39): Boundary refinement jumps to 7.0 — refine head actively correcting.
# Phase 3  (ep 40-59): IoU dominant — reduce span L1, amplify giou+boundary.
# Phase 4  (ep 60+  ): Maximum boundary precision (10.0) under cosine LR trough.
#
# Static cfg values above must match Phase 1 so build_criterion() starts correctly.
cfg["loss_schedule"] = [
    (0, {
        # v34: boundary refine starts at 4.0 (matches static cfg; zero-init is safe).
        # saliency/hyp_saliency are reduced globally via lw_saliency=0.05.
        "span_loss_coef":             10.0,
        "giou_loss_coef":              6.0,
        "boundary_refine_coef":        1.0,   # v34: 1.0→4.0; start high since head is zero-init
        "boundary_refine_giou_coef":   1.0,   # v34: 1.0→4.0; same
        "contrastive_align_loss_coef": 0.15,
        "loss_pop_coef":               0.1,
        "aux_loss_scale":              0.2,
        "set_cost_class":              2.0,
        "set_cost_span":              10.0,
        "set_cost_giou":               2.0,
    }),
    (15, {
      "boundary_refine_coef":        4.0,   # v34: 1.0→4.0; start high since head is zero-init
      "boundary_refine_giou_coef":   4.0,
    }),
    (20, {
        # Boundary refinement ramps to next tier — refine head now actively correcting
        "boundary_refine_coef":        7.0,   # v34: 4.0→7.0; next step since we start at 4.0
        "boundary_refine_giou_coef":   7.0,   # v34: same
        "aux_loss_scale":              0.3,
        "set_cost_class":              2.5,
        "set_cost_giou":               2.5,
    }),
    (40, {
        # IoU dominant — shift emphasis toward boundary precision
        "span_loss_coef":              8.0,
        "giou_loss_coef":              7.0,
        "boundary_refine_coef":        8.0,   # v34: 7.0→8.0
        "boundary_refine_giou_coef":   8.0,
        "contrastive_align_loss_coef": 0.05,
        "loss_pop_coef":               0.05,
        "set_cost_class":              3.0,
        "set_cost_span":               9.0,
        "set_cost_giou":               3.0,
    }),
    (60, {
        # Maximum boundary precision under cosine LR trough
        "span_loss_coef":              6.0,
        "giou_loss_coef":              8.0,
        "boundary_refine_coef":       10.0,
        "boundary_refine_giou_coef":  10.0,
        "set_cost_class":              3.5,
    }),
]


def get_cfg_defaults():
    os.makedirs(cfg["model_root"], exist_ok=True)
    with open(os.path.join(cfg["model_root"], "hyperparams.yaml"), "w") as f:
        yaml.dump(cfg, f)
    return cfg
