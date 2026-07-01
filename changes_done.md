# AVISM Changes Documentation

## Version History

| Version | Change | FSLA | HOTA | mAP |
|---------|--------|------|------|-----|
| Paper | Original | 42.78 | 61.73 | 40.57 |
| Baseline | Original code, our hardware | 39.32 | 61.11 | 40.23 |
| v2-SG+Reg+CBAM | SG Fusion (per-pixel) — **FAILED** | 1.04 | 10.89 | 6.29 |
| **v2.1-ConfGate+Reg+CBAM** | **Confidence Gate + Registry Tokens + CBAM1D** | TBD | TBD | TBD |

---

## Current Status: v2.1-ConfGate+Reg+CBAM

### What Went Wrong in v2 (Per-Pixel SG Fusion)
The SpatialGatedFusionLayer **replaced** the original cross-attention with per-pixel blending. This destroyed the targeted attention mechanism — instead of audio selectively querying visual features, it blended audio+visual at every pixel then averaged. Result: DetA dropped from 53.89 → 4.79, model couldn't detect anything.

### v2.1 Fix: Conservative Approach
Keep the **original cross-attention** (proven to work) but add enhancements on top:

### Module A: Audio Confidence Gate (Frame-Level) — NEW
- **What**: A learned sigmoid scalar that multiplies the cross-attention output
- **How**: `confidence = sigmoid(MLP(audio_feat))` → scales from 0 (silence) to 1 (informative)
- **Why**: Gives the model an "off switch" for audio in silent frames
- **Target**: FSLAn improvement (currently 15.15%)

### Module B: Registry Tokens (Video-Level) — KEPT from v2
- 4 learnable tokens on KV side of VL-AVFM cross-attention
- Absorb attention mass when audio is irrelevant

### Module C: CBAM1D (Video-Level) — KEPT from v2
- Channel + Temporal attention after AV fusion
- Refines fused features

### Files Changed

| File | Change |
|------|--------|
| [avism_transformer_decoder.py](file:///home/meeer/Documents/avis-project/avis/avism/modeling/transformer_decoder/avism_transformer_decoder.py) | Reverted to original cross-attention + added audio_confidence_gate |
| [avism.py](file:///home/meeer/Documents/avis-project/avis/avism/modeling/transformer_decoder/avism.py) | Registry tokens + CBAM1D (unchanged from v2) |
| [cbam.py](file:///home/meeer/Documents/avis-project/avis/avism/modeling/transformer_decoder/cbam.py) | CBAM2D + CBAM1D modules (unchanged) |
| [config.py](file:///home/meeer/Documents/avis-project/avis/avism/config.py) | NUM_REGISTERS = 4 (unchanged) |

---

## Training Command
```bash
conda activate avism
python train_net.py --config-file configs/avism/R50/avism_R50_IN.yaml
```

## Evaluation Command
```bash
python train_net.py --config-file configs/avism/R50/avism_R50_IN.yaml --eval-only MODEL.WEIGHTS outputs/avism_R50_IN_v2/model_final.pth
```
