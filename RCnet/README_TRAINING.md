# RCnet Training Pipeline

Documentation for training the Region Classifier (RCnet) with the improved pipeline:
FocalLoss alpha-balancing, best-model-on-val-F1 checkpointing, LR scheduler, and
dual STRICT/SOFT validation metrics.

## Prerequisites

### Conda environment

Use the `sat_env_training` conda environment (created from
`GNC-Payload/environment_training.yml`). Replace `$CONDA_PYTHON` below with the
path to your env's Python, e.g.:

```bash
CONDA_PYTHON=/path/to/miniconda3/envs/sat_env_training/bin/python
```

### Training data

Expected directory layout under `$DATA_DIR`:

```
$DATA_DIR/
├── 10S/
│   ├── l8_10S_00000.png
│   ├── l8_10S_00000_lat_lon.npz
│   ├── l8_10S_00000_mgrs_counts.json   # run count_mgrs_regions.py to generate
│   └── ...
├── 10T/
└── ...
```

You need one folder per MGRS region listed in `config.yaml` → `vision.salient_mgrs_region_ids`,
each containing images plus their `_mgrs_counts.json` label files.

### Repo layout assumption

Commands below assume you run from the repo root (the directory containing the
`RCnet/` folder). Set `$REPO` to that directory:

```bash
REPO=/path/to/Vision_Payload
cd $REPO
```

## Training

### Basic command

```bash
CUDA_VISIBLE_DEVICES=0 \
RC_ALPHA=0.9 RC_GAMMA=3 \
WANDB_MODE=offline \
PYTHONPATH=.:RCnet \
$CONDA_PYTHON RCnet/main.py \
  --train_flag --vram \
  --data_dir $DATA_DIR \
  --broken_files_path $REPO/broken_files.yaml \
  --vram_cache_path $REPO/RCnet/vram_cache \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --model_save_path $REPO/RCnet/output/model_my_run.pth \
  --save_plot_path $REPO/RCnet/output/loss_my_run.png \
  --save_plot_flag
```

### What happens

1. First run: loads all images from `$DATA_DIR` → VRAM, caches to `$REPO/RCnet/vram_cache/` (~35 min).
2. Subsequent runs: loads from cache in ~30 sec.
3. Every epoch: saves a checkpoint at `$REPO/RCnet/chkpts/model_fullvram_<epoch>.pth` and runs validation.
4. Best-F1 model is saved to `$REPO/RCnet/output/model_my_run_best.pth`.
5. After training: the best model is loaded and saved to `$REPO/RCnet/output/model_my_run.pth` (primary output).
6. `evaluate()` runs on the test set with STRICT metrics.

### Hyperparameters

| Knob | Where | Typical value | Effect |
|------|-------|---------------|--------|
| `RC_ALPHA` | env var | `0.9` | FocalLoss positive-class weight; higher → better recall (range 0.5–0.95) |
| `RC_GAMMA` | env var | `3` | FocalLoss focusing parameter; higher → more focus on hard examples |
| `--learning_rate` | CLI | `1e-4` | Adam learning rate. LR scheduler halves on val-F1 plateau. |
| `--batch_size` | CLI | `128` | Limited by VRAM; 128 fits easily in 24 GB for 224x224. |
| `--epochs` | CLI | `50` | Model converges well before 50; best checkpoint is preserved. |
| `CUDA_VISIBLE_DEVICES` | env var | `0` or `1` | Which GPU to use. |

### Validation metrics

Each epoch, two conventions are logged to wandb:

- **STRICT** (`validation_f1_strict`, `validation_precision_strict`, `validation_recall_strict`):
  a region counts as positive ground truth only if its calibrated label > 0.5 (roughly, the region covers ≥25% of the image).
- **SOFT** (`validation_f1_soft`, ...):
  any nonzero coverage counts as positive (old convention; inflates false negatives).

Best-model selection uses STRICT F1.

## Threshold sweep

After training, sweep the prediction threshold to find the best tradeoff between precision and recall.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
$CONDA_PYTHON $REPO/RCnet/threshold_sweep.py \
  --model $REPO/RCnet/output/model_my_run_best.pth \
  --size 224 224
```

Output is a table of F1/P/R for thresholds 0.05–0.5, under both STRICT and SOFT conventions.

## Per-class evaluation

Find the weakest regions (to inform data collection or further training).

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
$CONDA_PYTHON $REPO/RCnet/per_class_eval.py \
  --model $REPO/RCnet/output/model_my_run_best.pth \
  --strict_thresh 0.5 \
  --pred_thresh 0.5
```

Regions are listed in ascending F1 order (weakest first).

## Confusion analysis

Once you know a region is weak, find which other region it's confused with.

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$REPO \
$CONDA_PYTHON $REPO/RCnet/confusion_analysis.py \
  --model $REPO/RCnet/output/model_my_run_best.pth \
  --target_region 16T \
  --strict_thresh 0.5 \
  --pred_thresh 0.5
```

Shows false-positive co-occurrences (which regions are present when the model wrongly fires on the target) and false-negative substitutes (what the model predicts instead when it misses the target).

## Inspecting wandb metrics

Runs are saved offline under `$REPO/wandb/offline-run-<timestamp>-<id>/`.
To extract metrics without uploading, parse the .wandb datastore directly:

```python
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2

ds = datastore.DataStore()
ds.open_for_scan('<path_to_run>.wandb')
while True:
    data = ds.scan_data()
    if data is None: break
    rec = wandb_internal_pb2.Record()
    rec.ParseFromString(data)
    if rec.WhichOneof('record_type') == 'history':
        step = rec.history.step.num
        for item in rec.history.item:
            key = (list(item.nested_key) or [item.key])[0]
            if 'validation' in key:
                print(step, key, item.value_json)
```

## Tips

- **VRAM cache invalidation**: the cache key includes data split, region list, and image dimensions. Change any of those and a fresh cache is built.
- **LR scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` watches STRICT F1. If it plateaus for 3 epochs, LR is halved.
- **Alpha tuning**: start at 0.9 for high recall. Lower to 0.85 if over-predicting; raise toward 0.95 if recall still too low.
- **Saved models**: `$REPO/RCnet/output/MODEL_REGISTRY.md` and `model_registry.json` track hyperparameters and val metrics per experiment.
