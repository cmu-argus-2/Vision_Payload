#!/bin/bash
# Full retrain pipeline per region, on the GPU specified by CUDA_VISIBLE_DEVICES (default 0).
#
# For each region listed:
#   1. Ensure k-means 100-class catalog exists at $SRC/$REGION/bounding_boxes.csv
#   2. Symlink original training PNGs/npz into staging dir
#   3. Symlink bonus_images PNGs/npz with index_offset 500 (skips silently if no bonus dir)
#   4. prepare_yolo_data_gpu.py to generate labels
#   5. rotate_val_test_mirror.py with 8 fixed angles
#   6. train_yolo.py with degrees=360
#
# Usage: ./queue_full_retrain.sh [REGION ...]   (defaults to all 16)
# GPU:   set CUDA_VISIBLE_DEVICES (default 0)
#
# All sources are read-only via symlinks; only labels and rotated val/test
# images are real files. Re-running on a region is idempotent: existing
# symlinks are left in place, prep+mirror+train use --overwrite.

set -e

REPO=/home/sunac/Vision_Payload
PY=/home/argus/miniconda3/envs/sat_env_training/bin/python
LOGS=$REPO/LD/logs
SRC=/mnt/sda2/sunac_training_kmeans100
MIRROR=/mnt/sda2/sunac_training_kmeans100_rotval_v2
BONUS=/mnt/sda2/bonus_images/training
ORIG=/mnt/sda2/training
FIXED_ANGLES="0 45 90 135 180 -45 -90 -135"
GPU=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOGS"

ALL_REGIONS=(10S 10T 11R 12R 16T 17R 17T 18S 32S 32T 33S 33T 52S 53S 54S 54T)
REGIONS=("$@")
[ ${#REGIONS[@]} -eq 0 ] && REGIONS=("${ALL_REGIONS[@]}")

cp "$REPO/user_config.yaml" "$REPO/user_config.yaml.queue_full_bak"
restore_config() {
    cp "$REPO/user_config.yaml.queue_full_bak" "$REPO/user_config.yaml"
}
trap restore_config EXIT

echo "[$(date)] queue_full_retrain start: ${REGIONS[*]} on GPU $GPU" \
    | tee -a "$LOGS/queue_full_retrain.log"

for REGION in "${REGIONS[@]}"; do
    TAG="full_${REGION}"
    LOG="$LOGS/queue_${TAG}.log"
    echo "[$(date)] === $REGION === GPU $GPU" | tee -a "$LOG"

    STAGE_DIR="$SRC/$REGION"
    mkdir -p "$STAGE_DIR"

    # 1. k-means catalog
    if [ ! -f "$STAGE_DIR/bounding_boxes.csv" ]; then
        echo "[$(date)] generating k-means catalog" | tee -a "$LOG"
        FULL_CSV="$ORIG/$REGION/bounding_boxes.csv"
        [ -f "$FULL_CSV" ] || { echo "ERROR: no $FULL_CSV" | tee -a "$LOG"; exit 1; }
        PYTHONPATH=$REPO "$PY" "$REPO/LD/select_kmeans_landmarks.py" \
            --src "$FULL_CSV" --dst "$STAGE_DIR/bounding_boxes.csv" -k 100 \
            >> "$LOG" 2>&1
    fi

    # 2. symlink original PNGs (idempotent)
    for src_png in "$ORIG/$REGION"/l8_${REGION}_*.png; do
        [ -e "$src_png" ] || continue
        base=$(basename "$src_png")
        [ -e "$STAGE_DIR/$base" ] || ln -s "$src_png" "$STAGE_DIR/$base"
        npz="${src_png%.png}_lat_lon.npz"
        nbase=$(basename "$npz")
        [ -e "$npz" ] && [ ! -e "$STAGE_DIR/$nbase" ] && ln -s "$npz" "$STAGE_DIR/$nbase" || true
    done

    # 3. symlink bonus PNGs with index_offset 500
    BONUS_DIR="$BONUS/$REGION"
    if [ -d "$BONUS_DIR" ]; then
        for src_png in "$BONUS_DIR"/l8_${REGION}_*.png; do
            [ -e "$src_png" ] || continue
            base=$(basename "$src_png")
            idx=$(echo "$base" | sed -E "s/l8_${REGION}_0*([0-9]+)\\.png/\\1/")
            new_idx=$(printf "%05d" $((idx + 500)))
            new_base="l8_${REGION}_${new_idx}.png"
            [ -e "$STAGE_DIR/$new_base" ] || ln -s "$src_png" "$STAGE_DIR/$new_base"
            npz="${src_png%.png}_lat_lon.npz"
            new_npz="l8_${REGION}_${new_idx}_lat_lon.npz"
            [ -e "$npz" ] && [ ! -e "$STAGE_DIR/$new_npz" ] && ln -s "$npz" "$STAGE_DIR/$new_npz" || true
        done
    fi
    NPNG=$(ls "$STAGE_DIR"/l8_${REGION}_*.png 2>/dev/null | wc -l)
    echo "[$(date)] $REGION staging has $NPNG PNGs" | tee -a "$LOG"

    # 4. prep
    sed -i "s|^training_directory: .*$|training_directory: $SRC/|" "$REPO/user_config.yaml"
    cd "$REPO"
    echo "[$(date)] prep" | tee -a "$LOG"
    PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" LD/prepare_yolo_data_gpu.py \
        --regions "$REGION" --overwrite > "$LOGS/prep_${TAG}.log" 2>&1

    # 5. mirror with 8 fixed angles
    echo "[$(date)] rotate val/test mirror" | tee -a "$LOG"
    PYTHONPATH=$REPO "$PY" "$REPO/LD/rotate_val_test_mirror.py" \
        --regions "$REGION" \
        --src_root "$SRC" --out_root "$MIRROR" \
        --workers 16 --fixed_angles $FIXED_ANGLES \
        > "$LOGS/rotate_${TAG}.log" 2>&1

    # 6. train (degrees=360)
    sed -i "s|^training_directory: .*$|training_directory: $MIRROR/|" "$REPO/user_config.yaml"
    echo "[$(date)] train (degrees=360)" | tee -a "$LOG"
    PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$GPU "$PY" LD/train_yolo.py \
        --regions "$REGION" --degrees 360 --overwrite \
        > "$LOGS/train_${TAG}.log" 2>&1
    echo "[$(date)] $REGION done" | tee -a "$LOG"
done
echo "[$(date)] queue_full_retrain DONE" | tee -a "$LOGS/queue_full_retrain.log"
