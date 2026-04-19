# RCnet Model Registry

All models use:
- EfficientNet-B0 backbone with dropout=0.3
- AdamW optimizer, LR=1e-4, ReduceLROnPlateau scheduler (mode=max, factor=0.5, patience=3)
- 50 epochs training
- 16 MGRS regions: 10S, 10T, 11R, 12R, 16T, 17R, 17T, 18S, 32S, 32T, 33S, 33T, 52S, 53S, 54S, 54T
- 500 imgs/region (5600 train / 1200 val / 1200 test split 70/15/15)
- Input: 224x224
- Prediction threshold at inference: 0.5

## Results (validation set)

| Alpha | Gamma | Strict Thresh | Best Epoch | F1 | Precision | Recall | Model Path |
|-------|-------|---------------|------------|-----|-----------|--------|------------|
| 0.85 | 3 | 0.5 | 24 | 96.13 | 94.78 | 97.51 | `model_a085_strict_best.pth` |
| **0.90** | 3 | 0.5 | 24 | 96.12 | 94.21 | **98.10** | **`model_a090_strict_best.pth`** |
| 0.85 | 3 | 0.25 | 17 | 95.88 | 95.09 | 96.68 | `model_a085_s25_best.pth` |
| 0.90 | 3 | 0.25 | 30 | **96.34** | **95.83** | 96.87 | `model_a090_s25_best.pth` |

## Strict threshold interpretation

The strict threshold defines what counts as a "positive" ground-truth region:

- **strict=0.5** → region must cover ~25% of image (label > 0.5 under sigmoid calibration)
- **strict=0.25** → region must cover ~18% of image (label > 0.25)

## Recommendation

- **Highest recall**: `model_a090_strict_best.pth` (98.1% R — catches nearly all dominant regions)
- **Highest F1**: `model_a090_s25_best.pth` (96.3% F1 with 0.25 threshold)
- **Most balanced**: `model_a085_strict_best.pth` (best precision-recall tradeoff at strict=0.5)

See `model_registry.json` for programmatic access.
