# image_simulation

Off-nadir + color-tint simulation pipeline for evaluating the RC (region
classifier) and LD (landmark detector) networks on synthetic viewpoint and
chromatic perturbations of the validation set.

## What it does

1. **`run_offnadir_sim.py`** — for each image in the val/test split, reads its
   `_lat_lon.npz`, places a virtual satellite at `--altitude_km` directly above
   that point, renders the scene at each requested off-nadir roll angle by
   querying the GeoTIFF cache, and applies a set of color tints
   (`none`/`orange`/`purple` by default).
2. **`run_pipeline.py`** — walks the rendered PNG tree, runs the RC on each
   simulated image, then runs the region-specific YOLO LD on each predicted
   region. Emits a JSON with per-image RC correctness, LD detections, and
   aggregate recall/precision buckets by tint and angle.

Both scripts auto-detect CUDA via `torch.cuda.is_available()` and `ultralytics`;
no device flag is needed.

## Layout

```
image_simulation/
  run_offnadir_sim.py   # sim entrypoint
  run_pipeline.py       # RC + LD evaluation entrypoint
  png_simulator.py      # renders a single (angle, tint) view from a source tile
  camera_model.py       # camera intrinsics / extrinsics
  color_tint.py         # tint presets (none/orange/purple)
  blue_marble_*.py      # Blue Marble backdrop utilities
  earth_vis.py          # visualization helpers
  results/              # committed result JSONs/logs from prior runs
```

Generated artifacts (`sim_val100*/`, `sample_outputs*/`) are gitignored.

## Reproducing the hue=0.5 experiment

### 1. Train the RC with stronger hue jitter

Edit `RCnet/train_region_classifier_fullvram.py` and bump the `ColorJitter`
`hue` parameter from `0.15` to `0.5`, then train:

```bash
python RCnet/train_region_classifier_fullvram.py \
  --output_suffix a090_hue05 \
  # ... (other flags matching your baseline run)
```

Best-epoch checkpoint lands at
`RCnet/output/model_a090_hue05_best.pth`.

### 2. Simulate a validation set at 5-15° off-nadir

```bash
python image_simulation/run_offnadir_sim.py \
  --data_dir    /mnt/sda2/training \
  --output_dir  image_simulation/sim_val100_5to15 \
  --split       val \
  --num_images  100 \
  --altitude_km 590 \
  --angle_range 5 15 \
  --tints       none orange purple
```

Produces `sim_val100_5to15/<region>/<stem>/roll<angle>_<tint>.png`
(100 source images × 3 tints = 300 PNGs).

### 3. Run the pipeline

```bash
python image_simulation/run_pipeline.py \
  --sim_dir         image_simulation/sim_val100_5to15 \
  --rc_model        RCnet/output/model_a090_hue05_best.pth \
  --ld_models_base  /mnt/sda2/training \
  --source_data_dir /mnt/sda2/training \
  --output_json     image_simulation/results/pipeline_hue05_5to15.json \
  2>&1 | tee image_simulation/results/pipeline_hue05_5to15.log
```

## Results

From `results/pipeline_hue05_5to15.log` (RC = `model_a090_hue05_best.pth`,
sim = 100 val images × 3 tints, off-nadir 5-15°):

**RC (300 images):**

| slice  | n   | recall | precision |
| ------ | --- | ------ | --------- |
| all    | 300 | 87.0%  | 74.4%     |
| none   | 100 | 100.0% | 73.0%     |
| orange | 100 | 75.0%  | 74.3%     |
| purple | 100 | 86.0%  | 76.1%     |
| ~5°    | 72  | 84.7%  | 78.2%     |
| ~10°   | 165 | 87.3%  | 72.7%     |
| ~15°   | 63  | 88.9%  | 74.7%     |

- RC recall is the fraction of sim images whose predicted region set
  contains the source tile's MGRS region.
- RC precision is TP / total predicted regions (RC can emit multiple
  candidates per image; avg 1.17).
- Against the hue=0.15 baseline on a 5-30° sim (77.7% recall), the hue=0.5
  model reaches 87.0% on the 5-15° sim. Angle range differs, so not
  strictly apples-to-apples — but the color-robustness win is clear on the
  per-tint breakdown (purple recall 26.5% baseline → 86% here).

**LD end-to-end** (recall of baseline-detected landmarks, averaged per image):

| tint   | base_recall | gt_recall | gt_precision |
| ------ | ----------- | --------- | ------------ |
| none   | 55.8%       | 20.3%     | 95.4%        |
| orange | 46.9%       | 15.1%     | 96.1%        |
| purple | 26.5%       |  6.9%     | 100.0%       |

LD remains the pipeline bottleneck — precision is high but recall drops
sharply under tint, especially purple.

## Notes

- `--source_data_dir` must point at a directory laid out like
  `{region}/{stem}.png` and (optionally) `{region}/LD_training/val/labels/`
  with YOLO-format label `.txt`s for Level-2 (GT) metrics.
- PNG rendering uses `multiprocessing.Pool`; with large `--num_images` the
  output dir can grow fast (the 100×3 run above produces ~6 GB).
- Don't commit `sim_val100*/`, `sample_outputs*/`, or `RCnet/vram_cache*/` —
  they're gitignored.
