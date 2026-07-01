# AVISM Master Reference Document

> **Purpose**: Complete codebase reference for senior researcher-level understanding.

---

## Table of Contents
1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Components](#3-model-components)
4. [Loss Functions](#4-loss-functions)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Configuration System](#6-configuration-system)
7. [Key Modifications (Our Changes)](#7-key-modifications)

---

## 1. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AVISM FULL PIPELINE                              │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐     ┌─────────────┐
    │ Video Frames│     │ Audio (WAV) │
    │ [T, H, W, 3]│     │ [16kHz]     │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐     ┌─────────────┐
    │  Backbone   │     │  VGGish     │
    │  (ResNet50) │     │  (Frozen)   │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ▼                   ▼
    Multi-scale features  Audio features
    {res2,res3,res4,res5}    [T, 128]
           │                   │
           └─────────┬─────────┘
                     ▼
           ┌─────────────────┐
           │  MaskFormerHead │
           │  (sem_seg_head) │
           └────────┬────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   PixelDecoder         TransformerDecoder
   (FPN-style)          (Frame-Level)
        │                       │
        ▼                       ▼
   mask_features          frame_queries
   [T, C, H, W]           [L, T, fQ, C]
        │                       │
        └───────────┬───────────┘
                    ▼
           ┌─────────────────┐
           │   Avism Module  │
           │  (Video-Level)  │
           └────────┬────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   Predictions   │
           │ pred_logits     │
           │ pred_mask_embed │
           └─────────────────┘
```

---

## 2. Data Pipeline

### 2.1 Dataset Format (AVIS)
```
File: avism/data/datasets/avis.py
Format: Similar to YouTube-VIS

dataset/
├── train/
│   ├── JPEGImages/{video_id}/{frame_id}.jpg
│   └── audio/{video_id}.wav
├── test/
└── annotations/
    ├── train.json
    └── test.json
```

### 2.2 Data Mapper
```python
File: avism/data/dataset_mapper.py
Class: AVISDatasetMapper

Key operations:
1. Sample T frames from video (SAMPLING_FRAME_NUM, default=2)
2. Load audio features (VGGish pre-extracted, 128-dim)
3. Apply augmentations (flip, resize, crop)
4. Create instance masks and IDs
```

### 2.3 Audio Feature Extraction
```
File: avism/data/datasets/extract_audio_feat/audio_feature_extractor.py

VGGish Pipeline:
1. Resample to 16kHz
2. Compute mel spectrogram
3. Pass through VGGish network
4. Output: [T, 128] features per video
```

---

## 3. Model Components

### 3.1 AVISM Meta-Architecture
```python
File: avism/avism_model.py
Class: AVISM

Components:
├── backbone: ResNet-50/101 or Swin-L
├── sem_seg_head: MaskFormerHead
│   ├── pixel_decoder: MSDeformAttnPixelDecoder
│   └── predictor: AVISMMultiScaleMaskedTransformerDecoder
├── avism_module: Avism (video-level decoder)
├── criterion: SetCriterion (frame-level loss)
└── avism_criterion: AvismSetCriterion (clip-level loss)
```

### 3.2 Frame-Level Decoder
```python
File: avism/modeling/transformer_decoder/avism_transformer_decoder.py
Class: AVISMMultiScaleMaskedTransformerDecoder

Architecture:
├── input_proj: Project multi-scale features
├── pe_layer: PositionEmbeddingSine
├── level_embed: Learnable level embeddings
├── query_feat: Learnable query features [fQ, C]
├── query_embed: Learnable query positions [fQ, C]
├── av_gated_fusion: SpatialGatedFusionLayer (OUR MODIFICATION v3)
│   └── Audio-visual fusion with per-pixel gating
├── transformer_cross_attention_layers: [9 layers]
├── transformer_self_attention_layers: [9 layers]
├── transformer_ffn_layers: [9 layers]
├── class_embed: Linear(C, num_classes+1)
└── mask_embed: MLP(C, C, mask_dim, 3)

Forward Flow:
1. Project multi-scale features
2. Apply FL-AVFM (audio-visual fusion) → adds audio to queries
3. For each layer:
   - Cross-attention: queries ← image features
   - Self-attention: queries ← queries
   - FFN
4. Output: frame_queries, predictions
```

### 3.3 Video-Level Decoder
```python
File: avism/modeling/transformer_decoder/avism.py
Class: Avism

Architecture:
├── av_proj: Linear(128, C) for audio
├── enc_self_attn: Object Encoder (temporal self-attention)
├── enc_av_cross_attn: VL-AVFM (audio cross-attends to frame_query)
│   └── With ATTENTION SINKS (OUR MODIFICATION v4)
├── transformer_cross_attention_layers: Track Decoder
├── transformer_self_attention_layers: Track Decoder
├── transformer_ffn_layers: Track Decoder
├── query_feat: Video query features [cQ, C]
├── query_embed: Video query positions [cQ, C]
├── class_embed: Classification head
└── mask_embed: Mask embedding head

Forward Flow:
1. input_proj_dec(frame_query): Project frame queries
2. encode_frame_query(): Temporal self-attention with window attention
3. encode_av_fusion(): Audio cross-attends to frame_query (VL-AVFM)
   └── Registers added here (v4)
4. src = frame_query + av_feat
5. Track Decoder: video_queries cross-attend to src
6. Output: pred_logits, pred_mask_embed
```

### 3.4 Key Parameters
```yaml
HIDDEN_DIM: 256
NUM_OBJECT_QUERIES: 100 (frame), 100 (video)
ENC_LAYERS: 6 (video-level)
DEC_LAYERS: 3 (video-level)
ENC_WINDOW_SIZE: 6 (temporal window)
NHEADS: 8
DIM_FEEDFORWARD: 2048
```

---

## 4. Loss Functions

### 4.1 Frame-Level Loss (SetCriterion)
```python
File: mask2former/modeling/criterion.py

Losses:
├── loss_ce: Cross-entropy for classification
├── loss_mask: Binary cross-entropy for masks (point sampling)
└── loss_dice: Dice loss for mask overlap
```

### 4.2 Clip-Level Loss (AvismSetCriterion)
```python
File: avism/modeling/avism_criterion.py
Class: AvismSetCriterion

Losses:
├── loss_avism_ce: Cross-entropy (clip-level)
├── loss_avism_mask: BCE mask loss (clip-level)
├── loss_avism_dice: Dice loss (clip-level)
└── loss_avism_sim: Foreground similarity loss (contrastive)

Matching: AvismHungarianMatcher
├── cost_class: Classification cost
├── cost_mask: BCE mask cost
└── cost_dice: Dice cost
```

---

## 5. Evaluation Metrics

### 5.1 HOTA (Higher Order Tracking Accuracy)
```python
File: avism/data/aviseval/metrics/hota.py

HOTA = sqrt(DetA × AssA)

Components:
├── DetA = TP / (TP + FN + FP)  # Detection accuracy
├── AssA = Association accuracy  # Track consistency
├── DetRe = TP / (TP + FN)       # Detection recall
├── DetPr = TP / (TP + FP)       # Detection precision
├── AssRe = Association recall
├── AssPr = Association precision
└── LocA = Mean IoU of matches

Computed at alpha thresholds: 0.05 to 0.95
Final score: Mean across alphas
```

### 5.2 FSLA (Frame-level Sounding Localization Accuracy)
```python
File: avism/data/aviseval/metrics/av_loc.py

FA = (n_tp + s_tp + m_tp) / total_frames

Sub-metrics:
├── FAn: Accuracy on NO sound frames
├── FAs: Accuracy on SINGLE sound source frames
└── FAm: Accuracy on MULTIPLE sound source frames

Match criteria:
1. Same class set (predicted == ground truth)
2. Same instance count
3. Per-instance IoU > alpha
```

### 5.3 mAP (Track-level AP)
```python
File: avism/data/aviseval/metrics/track_map.py

Standard COCO-style AP computed per track.
Area ranges: small, medium, large
```

### 5.4 Evaluation Pipeline
```python
File: avism/data/avis_eval.py
Class: AVISEvaluator

Flow:
1. process(): Convert predictions to COCO format
2. evaluate(): Run aviseval metrics
3. Output: segm dict with all metrics
```

---

## 6. Configuration System

### 6.1 Config Files
```
configs/
├── avism/
│   ├── Base-AVIS.yaml        # Base config (MODIFIED for 1 GPU)
│   └── R50/
│       └── avism_R50_IN.yaml  # ResNet-50 + ImageNet config
```

### 6.2 Key Config Locations
```python
# avism/config.py - AVISM-specific configs
cfg.MODEL.AVISM.HIDDEN_DIM = 256
cfg.MODEL.AVISM.NUM_OBJECT_QUERIES = 100
cfg.MODEL.AVISM.ENC_LAYERS = 6
cfg.MODEL.AVISM.ENC_WINDOW_SIZE = 6
cfg.MODEL.AVISM.NUM_REGISTERS = 4  # OUR ADDITION

# mask2former/config.py - MaskFormer configs
cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100
cfg.MODEL.MASK_FORMER.DEC_LAYERS = 10
```

---

## 7. Key Modifications (Our Changes)

### v3: SpatialGatedFusionLayer (FL-AVFM)
```python
File: avism_transformer_decoder.py, lines 137-245

Purpose: Per-pixel audio-visual fusion with gate mechanism

Gate formula:
  gate = sigmoid(W_g @ [audio; video])
  output = gate * audio_proj + (1-gate) * video_proj

Gate-weighted pooling:
  importance = gate.mean(dim=-1)
  fused = weighted_sum(spatial_fused * importance)
```

### v4: Attention Sinks (VL-AVFM)
```python
File: avism.py, modified methods:
- __init__: Added register_tokens [4, 256]
- encode_av_fusion: Concat registers to frame_query
- _window_av_attn: Concat registers + extend mask
- _shift_window_av_attn: Concat registers + extend mask

Purpose: Provide "sink" tokens for audio attention when no visual match
```

### Training Infrastructure
```python
File: train_net.py

GradientAccumulationTrainer:
- Simulates 2-GPU batch on 1 GPU
- Accumulates gradients over 2 steps
- Proper LR scheduler stepping

Config changes (Base-AVIS.yaml):
- GRADIENT_ACCUMULATION_STEPS: 2
- MAX_ITER: 96000 (doubled)
- STEPS: (64000,) (doubled)
```

---

## Quick Reference: File → Component

| File | Component | Affects Metric |
|------|-----------|----------------|
| `avism_transformer_decoder.py` | FL-AVFM, Frame Decoder | FSLA, mAP |
| `avism.py` | Object Encoder, VL-AVFM, Track Decoder | HOTA, FSLA |
| `avism_model.py` | Meta-architecture, Inference | All |
| `avism_criterion.py` | Clip-level losses | Training |
| `avism_matcher.py` | Hungarian matching | Training |
| `hota.py` | HOTA metric | Evaluation |
| `av_loc.py` | FSLA metric | Evaluation |
| `avis_eval.py` | Evaluator pipeline | Evaluation |
