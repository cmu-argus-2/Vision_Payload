# RCnet - Region Classifier Network

RCnet is a deep learning-based multi-label image classifier that uses EfficientNet-B0 to classify satellite images based on their geographic regions (MGRS zones). The classifier predicts which MGRS regions are visible in each image using multi-hot encoded labels.

## Table of Contents
- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Possible Problems](#possible-problems)
- [CuPy Installation (CUDA 12.x)](#cupy-installation-cuda-12x)

---

## Overview

The RCnet pipeline consists of two main stages:
1. **Data Preparation**: Process geotagged satellite images to count MGRS region occurrences
2. **Training**: Train the EfficientNet-B0 model to classify images by region

---

## Data Preparation

The data preparation step processes geotagged satellite images and generates MGRS region count files that are used as labels during training.

### Prerequisites

Ensure your training directory is structured as follows:
```
/training_directory/
├── {region_1}/
│   ├── 00000.png
│   ├── 00000_lat_lon.npz
│   ├── 00001.png
│   ├── 00001_lat_lon.npz
│   └── ...
├── {region_2}/
│   └── ...
└── ...
```

Each image must have a corresponding `_lat_lon.npz` file containing latitude/longitude coordinates for each pixel.

### Running Data Preparation

Execute the MGRS region counting script:

```bash
python RCnet/count_mgrs_regions.py \
    --regions 17R 18R 19R 20R 21R \
    --num_processes 8 \
    --use_gpu
```

**Arguments:**
- `--regions`: List of MGRS regions to count (default: loaded from config)
- `--num_processes`: Number of parallel processes to use (default: CPU count)
- `--use_gpu`: Enable GPU acceleration with CuPy (recommended)
- `--overwrite`: Overwrite existing MGRS count files
- `--resume`: Skip images that already have count files

**Output:**
The script generates `*_mgrs_counts.json` files alongside each image:
```
/training_directory/
├── {region_1}/
│   ├── 00000.png
│   ├── 00000_lat_lon.npz
│   ├── 00000_mgrs_counts.json  ← Generated
│   └── ...
```

**Note:** GPU acceleration via CuPy significantly speeds up processing. See [CuPy Installation](#cupy-installation-cuda-12x) section below.

---

## Training

Once data preparation is complete, train the region classifier model.

### Basic Training Command

```bash
python RCnet/main.py \
    --train_flag \
    --data_dir /path/to/training_directory \
    --model_save_path models/rcnet_model.pth \
    --epochs 50 \
    --learning_rate 1e-3
```

### Training with Non-Salient Data

To improve model robustness, you can include non-salient (background) images:

```bash
python RCnet/main.py \
    --train_flag \
    --data_dir /path/to/training_directory \
    --non_salient_data_dir /path/to/non_salient_images \
    --model_save_path models/rcnet_model.pth \
    --epochs 50 \
    --learning_rate 1e-3 \
    --save_plot_flag \
    --save_plot_path results/training_plot.png
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train_flag` | Enable training mode | (flag) |
| `--data_dir` | Path to training directory (required) | - |
| `--non_salient_data_dir` | Path to non-salient images (optional) | None |
| `--model_save_path` | Path to save trained model | `model.pth` |
| `--model_load_path` | Path to load pre-trained model | `model.pth` |
| `--epochs` | Number of training epochs | 10 |
| `--learning_rate` | Learning rate for optimizer | 1e-3 |
| `--save_plot_flag` | Save training loss plots | (flag) |
| `--save_plot_path` | Path to save training plot | `plot.png` |

### Data Splits

The dataset is automatically split:
- **Training**: 70%
- **Validation**: 15%
- **Testing**: 15%

### Model Evaluation

To evaluate a trained model without training:

```bash
python RCnet/main.py \
    --data_dir /path/to/training_directory \
    --model_load_path models/rcnet_model.pth
```

Evaluation metrics are computed on the test set and logged to the console.

---

## Possible Problems

### 1. **Missing MGRS Count Files**
**Error:** `JSON file not found for {image_path}. Skipping this image.`

**Solution:** Run the data preparation step first:
```bash
python RCnet/count_mgrs_regions.py --use_gpu
```

### 2. **CUDA Out of Memory**
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size in [train_region_classifier.py](train_region_classifier.py#L175) (default: 128)
- Clear CUDA cache before training (already implemented in main.py)
- Use a GPU with more memory

### 3. **Number of Classes Mismatch**
**Error:** `AssertionError: Number of classes mismatch!`

**Solution:** Ensure the number of regions in your training data matches `RegionClassifier.NUM_CLASSES`. Update the configuration file or retrain with consistent regions.

### 4. **CuPy Not Available During Data Prep**
**Warning:** `Warning: CuPy not available. Falling back to CPU computation.`

**Solution:** Install CuPy for GPU acceleration (see section below). CPU processing will work but is significantly slower.

### 5. **File Permissions / I/O Errors**
**Error:** `PermissionError` or slow data loading

**Solutions:**
- Ensure read/write permissions for training directory
- If using HDD, reduce `num_workers` in DataLoader (default: 0)
- Consider moving data to SSD for faster I/O

### 6. **Malformed Image or Lat/Lon Files**
**Error:** Image loading errors during data preparation

**Solution:** The `GeotaggedImage.load()` function automatically detects and deletes malformed files if `delete_bad_files=True` is set.

### 7. **Training Not Converging**
**Solutions:**
- Increase number of epochs (try 50-100)
- Adjust learning rate (try 1e-4 or 1e-2)
- Check data augmentation settings in [train_region_classifier.py](train_region_classifier.py#L135-L163)
- Ensure sufficient training data for each class

---

## CuPy Installation (CUDA 12.x)

CuPy provides GPU acceleration for MGRS region counting, significantly improving data preparation speed.

### Installation for CUDA 12.x

For CUDA 12.x versions, install the appropriate CuPy package:

```bash
# CUDA 12.0
pip install cupy-cuda12x

# Or specify exact CUDA version
pip install cupy-cuda120  # CUDA 12.0
pip install cupy-cuda121  # CUDA 12.1
pip install cupy-cuda122  # CUDA 12.2
```

### Verify Installation

```bash
python -c "import cupy as cp; print(cp.__version__); print('CUDA available:', cp.cuda.is_available())"
```

Expected output:
```
12.x.x
CUDA available: True
```

### CUDA Version Compatibility

| CUDA Version | CuPy Package |
|--------------|--------------|
| 12.0 | `cupy-cuda120` |
| 12.1 | `cupy-cuda121` |
| 12.2+ | `cupy-cuda12x` |
| 11.x | `cupy-cuda11x` |

### Check Your CUDA Version

```bash
nvcc --version
# or
nvidia-smi
```

### Installation Troubleshooting

**If CuPy installation fails:**
1. Verify CUDA toolkit is installed: `nvcc --version`
2. Ensure compatible GCC/compiler version
3. Try installing from conda: `conda install -c conda-forge cupy`
4. As a fallback, CPU computation will be used (slower but functional)

**Performance Impact:**
- With CuPy (GPU): ~10-100x faster for large datasets
- Without CuPy (CPU): Still functional, but data prep may take hours

---

## Additional Notes

- The model uses **EfficientNet-B0** architecture with a custom classifier head
- Labels are multi-hot encoded with sigmoid-transformed region counts
- Training integrates with **Weights & Biases (wandb)** for experiment tracking
- Image size: 224x224 pixels (automatically resized)
- Data augmentation includes color jittering, normalization, and optional transforms

For more details, see the docstrings in:
- [train_region_classifier.py](train_region_classifier.py)
- [dataloader.py](dataloader.py)
- [count_mgrs_regions.py](count_mgrs_regions.py)
