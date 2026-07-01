# Future Contributions: SeaVIS Techniques for AVISM Pipeline

**Source:** SeaVIS (arXiv: 2603.01431, 2026)  
**Expected Impact:** +2.0 FSLA, +1.6 HOTA, +0.4 mAP (from CCAF) + additional +3.97 FSLA (from AGCL)

---

## Contribution 1: Causal Cross-Attention Fusion (CCAF)

### What
Replace the current "in-frame" audio-visual fusion (`av_sf`) with temporal causal cross-attention. Visual features (query) attend to the **entire audio history** (key/value) with a causal mask.

### Why
Current `av_sf` fuses each frame's audio independently — frame 3 has zero knowledge of audio from frames 1-2. CCAF lets each frame leverage temporal audio context.
 
### Where to change
- **`avism_transformer_decoder.py`**
  - Add `USE_CCAF` flag via `from_config()` + `cfg.MODEL.AVISM.USE_CCAF`
  - Add CCAF module in `__init__()` (3 cross-attention layers + learnable audio temporal pos encoding)
  - Branch in `forward()`: if `use_ccaf`, apply CCAF to `src` list; else use existing `av_sf`
- **`config.py`**
  - Add `cfg.MODEL.AVISM.USE_CCAF = False` (default off for backward compat)

### CCAF Logic (pseudocode)
```python
# In forward(), after building src list (with CBAM + SGF applied):

# Prepare audio KV: [BT, C] → [T, B, C]  
audio_kv = audio_proj.view(B, T, C).permute(1, 0, 2) + audio_temporal_pos[:T]

for each scale i:
    # Reshape visual: [H*W, BT, C] → [T*H*W, B, C]
    src_reshaped = reshape(src[i])
    
    # Causal mask: [T*H*W, T], block future audio
    mask[j, k] = True if (j // HW) < k  
    
    # Cross-attention: visual attends to audio history
    enhanced = cross_attn(query=src_reshaped, key=audio_kv, value=audio_kv, mask=mask)
    src[i] = reshape_back(src_reshaped + enhanced)

# Simple audio injection into queries (replaces av_sf + av_post_proj)
output = query_feat + audio_proj.expand(num_queries, BT, C)
```

### Backward Compatibility
- `USE_CCAF = False` → uses existing `av_sf` path → old weights load fine for eval
- `USE_CCAF = True` → uses CCAF path → new training only

---

## Contribution 2: Two-Level Audio-Guided Contrastive Learning (AGCL)

### What
Train instance embeddings to encode both visual appearance AND sounding activity via contrastive learning at **frame-level** and **instance-level**.

### Why
- Frame-level alone distinguishes sounding vs silent objects within a frame, but **HURTS mAP/HOTA** (SeaVIS ablation: mAP drops 39.85 vs 40.03)
- Instance-level distinguishes sounding vs silent **states of the same object** across frames
- Together they're complementary and give the best results

### Where to change

#### Frame-Level (already implemented, just enable)
- **`config.py`**: Set `AGCL_FRAME_WEIGHT = 1.0`
- **`avism_criterion.py`**: `loss_frame_contrastive()` already exists (lines 122-207)
- **`avism_model.py`**: Audio projection for AGCL already exists (lines 299-304)

#### Instance-Level (new, needs implementation)
- **`config.py`**
  - Add `cfg.MODEL.AVISM.AGCL_INSTANCE_WEIGHT = 1.0`
- **`avism_criterion.py`**
  - Add `loss_instance_contrastive()` method
  - Register in `loss_map` as `'agcl_instance'`
- **`avism_model.py`**
  - Add `loss_agcl_instance` to weight dict when instance weight > 0

### Instance-Level AGCL Logic (pseudocode)
```python
def loss_instance_contrastive(outputs, clip_targets, frame_targets, 
                               clip_indices, frame_indices, num_masks):
    """
    For each tracked instance k across all T frames:
      1. Compute instance-specific audio anchor:
         ā_k = mean(audio_anchors from frames where k is sounding)
      2. Positive set: k's embeddings from SOUNDING frames
      3. Negative set: k's embeddings from SILENT frames
      4. Multi-positive InfoNCE loss
    """
    fq = outputs["pred_fq_embed"][-1]  # [B, T, fQ, C]
    audio_anchors = outputs["audio_feats_proj"]  # [B, T, C]
    
    for each instance k matched across frames:
        sounding_frames = frames where instance k has GT mask
        silent_frames = frames where instance k exists but no GT mask
        
        if len(sounding_frames) == 0 or len(silent_frames) == 0:
            continue  # Need both states
        
        # Instance-specific audio anchor
        anchor_k = mean(audio_anchors[sounding_frames])  # [C]
        anchor_k = L2_normalize(anchor_k)
        
        # Positive: embeddings from sounding frames
        pos_embeds = L2_normalize(fq[k, sounding_frames])  # [N_pos, C]
        # Negative: embeddings from silent frames  
        neg_embeds = L2_normalize(fq[k, silent_frames])  # [N_neg, C]
        
        # Multi-positive InfoNCE
        pos_sims = pos_embeds @ anchor_k / tau  # [N_pos]
        neg_sims = neg_embeds @ anchor_k / tau  # [N_neg]
        
        for each positive p in pos_sims:
            logits = cat([p, neg_sims])
            loss += cross_entropy(logits, target=0)
    
    return {"loss_agcl_instance": loss / num_valid}
```

---

## Implementation Order

1. **First:** Implement CCAF (medium effort, high impact, low risk)
2. **Then:** Enable both AGCL levels together (MUST be together — frame-level alone hurts)
3. **Both behind config flags** for backward compatibility

## Config Flags Summary
```python
# config.py additions:
cfg.MODEL.AVISM.USE_CCAF = False              # Enable CCAF temporal fusion
cfg.MODEL.AVISM.AGCL_FRAME_WEIGHT = 0.0       # Already exists, set to 1.0 to enable
cfg.MODEL.AVISM.AGCL_INSTANCE_WEIGHT = 0.0    # New, set to 1.0 to enable
cfg.MODEL.AVISM.AGCL_TEMPERATURE = 0.07       # Already exists
```
