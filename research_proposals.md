# AVISM Research Enhancement Proposals

> **Goal**: Exceed paper results (FSLA 42.78, HOTA 61.73, mAP 40.57)
> **Current**: FSLA 39.82, HOTA 58.45, mAP 35.95

---

## Priority Matrix

| Enhancement | Impact | Difficulty | Novelty | Priority |
|-------------|--------|------------|---------|----------|
| Modern Audio Encoder | ⭐⭐⭐⭐⭐ | Medium | ⭐⭐⭐ | **P0** |
| Temporal Consistency Loss | ⭐⭐⭐⭐ | Low | ⭐⭐ | **P0** |
| Audio-Visual Contrastive Learning | ⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐ | **P1** |
| Multi-Scale Audio Features | ⭐⭐⭐ | Medium | ⭐⭐⭐ | **P1** |
| Query Refinement | ⭐⭐⭐ | Low | ⭐⭐ | **P2** |
| Audio-Guided Deformable Attention | ⭐⭐⭐⭐ | High | ⭐⭐⭐⭐⭐ | **P2** |

---

## 1. Audio Encoder Enhancement [HIGH IMPACT]

### 1.1 Replace VGGish with Modern Audio Encoder

**Problem**: VGGish is from 2017, produces only 128-dim features, frozen.

**Options**:

| Model | Dim | Pre-training | Pros |
|-------|-----|--------------|------|
| **BEATs** | 768 | AudioSet | SOTA audio encoder, transformer-based |
| **AudioMAE** | 768 | AudioSet | Self-supervised, good generalization |
| **AST** | 768 | AudioSet | Audio Spectrogram Transformer |
| **CLAP** | 512 | AudioSet+text | Text-audio alignment, zero-shot |

**Recommended**: BEATs (best performance) or CLAP (enables zero-shot)

**Implementation**:
```python
# In avism/modeling/audio_encoder.py (NEW FILE)
class BEATsEncoder(nn.Module):
    def __init__(self, pretrained_path, hidden_dim=256):
        self.beats = BEATs.load(pretrained_path)
        self.proj = nn.Linear(768, hidden_dim)
    
    def forward(self, audio):
        # audio: [B, T, samples]
        feats = self.beats(audio)  # [B, T, 768]
        return self.proj(feats)    # [B, T, 256]
```

**Expected Impact**: +3-5% on all metrics (richer audio representations)

---

### 1.2 Multi-Scale Audio Features

**Problem**: Current audio is single-scale [T, 128]. Misses temporal details.

**Solution**: Extract audio at multiple temporal scales.

```python
# Multi-scale audio extraction
audio_1s = extract_features(audio, window=1.0)   # Coarse
audio_500ms = extract_features(audio, window=0.5)  # Medium
audio_250ms = extract_features(audio, window=0.25) # Fine

# Fuse in VL-AVFM
audio_multi = self.audio_fusion([audio_1s, audio_500ms, audio_250ms])
```

**Expected Impact**: +1-2% FSLA (better temporal localization)

---

## 2. FL-AVFM Enhancement [MEDIUM IMPACT]

### 2.1 Audio-Guided Deformable Attention

**Problem**: Current gating is per-pixel but doesn't adaptively sample.

**Solution**: Use audio to guide WHERE to look in the image.

```python
class AudioGuidedDeformableAttention(nn.Module):
    def forward(self, visual, audio):
        # Audio predicts sampling offsets
        offsets = self.offset_net(audio)  # [B, K, 2]
        
        # Sample visual features at predicted locations
        sampled = deformable_sample(visual, offsets)
        
        # Fuse
        return self.fusion(sampled, audio)
```

**Expected Impact**: +2-3% FSLA (better spatial localization)
**Novelty**: High - audio guides spatial attention

---

### 2.2 Class-Aware Audio Embeddings

**Problem**: Audio features are class-agnostic.

**Solution**: Learn class-specific audio prototypes.

```python
# Learnable audio prototypes per class
self.audio_prototypes = nn.Embedding(num_classes, hidden_dim)

# Match audio to class prototypes
similarity = F.cosine_similarity(audio, self.audio_prototypes.weight)
class_weights = F.softmax(similarity, dim=-1)

# Weight fusion by class relevance
fused = (class_weights @ self.audio_prototypes.weight) + audio
```

**Expected Impact**: +1-2% mAP (better class discrimination)

---

## 3. VL-AVFM Enhancement [MEDIUM IMPACT]

### 3.1 Fix Attention Sinks (Current Issue)

**Problem**: Registers absorb too much attention.

**Solutions**:

**Option A**: Reduce registers
```python
NUM_REGISTERS = 2  # Instead of 4
```

**Option B**: Regularization loss
```python
# Penalize high attention to registers
reg_attn = attention_weights[:, :, -num_reg:]
loss_reg = 0.1 * reg_attn.mean()
```

**Option C**: Sparse attention to registers
```python
# Only allow attention to registers when confidence is low
mask_registers = confidence < 0.3
```

---

### 3.2 Audio-Visual Memory Bank

**Problem**: Window attention limits long-range dependencies.

**Solution**: Maintain memory bank of past audio-visual features.

```python
class AVMemoryBank(nn.Module):
    def __init__(self, memory_size=100):
        self.memory = None  # [memory_size, C]
    
    def update(self, features, scores):
        # Keep top-K features based on confidence
        top_k = self.select_top_k(features, scores)
        self.memory = torch.cat([self.memory, top_k])[-self.size:]
    
    def retrieve(self, query):
        # Cross-attention to memory
        return self.attention(query, self.memory, self.memory)
```

**Expected Impact**: +1-2% HOTA (better long-range tracking)

---

## 4. Loss Function Enhancement [HIGH IMPACT, LOW DIFFICULTY]

### 4.1 Temporal Consistency Loss ⭐ RECOMMENDED

**Problem**: Predictions can be inconsistent across frames.

**Solution**: Enforce smooth predictions over time.

```python
def temporal_consistency_loss(pred_masks, pred_logits):
    # Mask consistency: masks should change smoothly
    mask_diff = (pred_masks[:, 1:] - pred_masks[:, :-1]).abs()
    loss_mask = mask_diff.mean()
    
    # Logit consistency: classes shouldn't flicker
    logit_diff = (pred_logits[:, 1:] - pred_logits[:, :-1]).abs()
    loss_logit = logit_diff.mean()
    
    return 0.1 * loss_mask + 0.1 * loss_logit
```

**Expected Impact**: +2-3% HOTA (smoother tracking)
**Implementation**: Very easy - just add to loss

---

### 4.2 Audio-Visual Alignment Loss

**Problem**: No explicit supervision for audio-visual correspondence.

**Solution**: Contrastive loss between audio and visual features.

```python
def av_contrastive_loss(audio_feat, visual_feat, labels):
    # Same object should have similar audio-visual features
    similarity = F.cosine_similarity(audio_feat, visual_feat)
    
    # Positive pairs: same instance
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # InfoNCE loss
    loss = -torch.log(
        torch.exp(similarity[pos_mask]).sum() / 
        torch.exp(similarity).sum()
    )
    return loss
```

**Expected Impact**: +2-3% FSLA (better audio-visual alignment)

---

### 4.3 Track Association Loss

**Problem**: AssA (association accuracy) could be improved.

**Solution**: Explicitly supervise track consistency.

```python
def track_association_loss(query_embeds, gt_ids):
    # Same track ID should have similar embeddings across frames
    for track_id in gt_ids.unique():
        track_mask = gt_ids == track_id
        track_embeds = query_embeds[track_mask]
        
        # Minimize variance within track
        loss += track_embeds.var(dim=0).mean()
    
    return loss
```

**Expected Impact**: +1-2% HOTA (better AssA)

---

## 5. Object Encoder Enhancement [MEDIUM IMPACT]

### 5.1 Hierarchical Temporal Modeling

**Problem**: Flat temporal attention misses multi-scale patterns.

**Solution**: Pyramid temporal attention.

```python
# Temporal pyramid
feat_2 = pool_temporal(feat, kernel=2)   # /2 temporal
feat_4 = pool_temporal(feat, kernel=4)   # /4 temporal

# Multi-scale self-attention
out_1 = self.attn_1(feat, feat, feat)
out_2 = self.attn_2(feat_2, feat_2, feat_2)
out_4 = self.attn_4(feat_4, feat_4, feat_4)

# Upsample and fuse
output = out_1 + upsample(out_2) + upsample(out_4)
```

**Expected Impact**: +1-2% HOTA

---

### 5.2 Learnable Temporal Position Encoding

**Problem**: Current temporal encoding may be suboptimal.

**Solution**: Use learnable or sinusoidal+learnable hybrid.

```python
self.temporal_embed = nn.Embedding(max_frames, hidden_dim)
self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)

# Combine fixed + learnable
pos = self.sinusoidal_pos(t) + self.temporal_embed(t)
pos = self.temporal_proj(pos)
```

---

## 6. Data & Training Enhancement [LOW HANGING FRUIT]

### 6.1 Audio Augmentation

**Current**: No audio augmentation.

**Add**:
```python
# In dataset_mapper.py
if self.is_train:
    audio = self.time_stretch(audio, rate=random.uniform(0.9, 1.1))
    audio = self.pitch_shift(audio, semitones=random.uniform(-2, 2))
    audio = self.add_noise(audio, snr=random.uniform(20, 40))
```

**Expected Impact**: +1% on all metrics

---

### 6.2 Audio-Visual Sync Augmentation

**Problem**: Audio-visual sync is always perfect.

**Solution**: Randomly shift audio timing.

```python
# Shift audio by ±0.5s randomly
shift = random.uniform(-0.5, 0.5)
audio = shift_audio(audio, shift_seconds=shift)
```

**Expected Impact**: +0.5-1% robustness

---

### 6.3 MixUp for Audio-Visual

**Solution**: Mix audio-visual pairs.

```python
lambda_ = np.random.beta(0.4, 0.4)
mixed_video = lambda_ * video1 + (1-lambda_) * video2
mixed_audio = lambda_ * audio1 + (1-lambda_) * audio2
mixed_labels = lambda_ * labels1 + (1-lambda_) * labels2
```

---

## 7. Inference Enhancement [QUICK WINS]

### 7.1 Test-Time Augmentation

```python
def inference_tta(model, video, audio):
    pred1 = model(video, audio)
    pred2 = model(flip(video), audio)
    pred3 = model(scale(video, 0.9), audio)
    
    return ensemble([pred1, flip_back(pred2), resize_back(pred3)])
```

**Expected Impact**: +0.5-1% on all metrics

---

### 7.2 Confidence Calibration

**Problem**: Confidence threshold (0.3) may not be optimal.

**Solution**: Learn per-class thresholds on validation set.

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days each)
1. ✅ Fix attention sinks (reduce to 2)
2. ⬜ Add temporal consistency loss
3. ⬜ Add audio augmentation

### Phase 2: Medium Effort (3-5 days each)
4. ⬜ Modern audio encoder (BEATs)
5. ⬜ Audio-visual contrastive loss
6. ⬜ Track association loss

### Phase 3: High Effort (1-2 weeks each)
7. ⬜ Audio-guided deformable attention
8. ⬜ Multi-scale audio features
9. ⬜ Audio-visual memory bank

---

## Expected Combined Impact

| Metric | Current | +Phase1 | +Phase2 | +Phase3 |
|--------|---------|---------|---------|---------|
| FSLA | 39.82 | 41-42 | 43-45 | 45-48 |
| HOTA | 58.45 | 60-61 | 62-64 | 65+ |
| mAP | 35.95 | 37-38 | 40-42 | 43+ |
