#!/bin/bash
# Convert all unconverted bonus GeoTIFFs to PNG+lat_lon.npz pairs in
# /mnt/sda2/bonus_images/training/{region}/ so future retrains can symlink them.
# CPU-only, ~5 min per region with 32 workers, runs sequentially.

set -e

REPO=/home/sunac/Vision_Payload
PY=/home/argus/miniconda3/envs/sat_env_training/bin/python
LOGS=$REPO/LD/logs
mkdir -p "$LOGS"

REGIONS=("17T" "32S" "32T" "33S" "33T" "53S")
for region in "${REGIONS[@]}"; do
    SRC_DIR="/mnt/sda2/bonus_images/geotiffs/$region"
    OUT_DIR="/mnt/sda2/bonus_images/training/$region"
    if [ ! -d "$SRC_DIR" ]; then
        echo "[$(date)] $region: no geotiffs source, skipping"
        continue
    fi
    mkdir -p "$OUT_DIR"
    echo "[$(date)] $region: converting $(ls $SRC_DIR/*.tif 2>/dev/null | wc -l) tifs -> $OUT_DIR" \
        | tee -a "$LOGS/convert_all_bonus.log"
    "$PY" "$REPO/LD/convert_bonus_tifs.py" \
        --src_dir "$SRC_DIR" \
        --out_dir "$OUT_DIR" \
        --region "$region" \
        --src_prefix "l8_${region}_" \
        --index_offset 0 \
        > "$LOGS/convert_bonus_${region}.log" 2>&1
    echo "[$(date)] $region: done. PNGs in dst: $(ls $OUT_DIR/*.png 2>/dev/null | wc -l)" \
        | tee -a "$LOGS/convert_all_bonus.log"
done
echo "[$(date)] CONVERT_ALL_BONUS complete." | tee -a "$LOGS/convert_all_bonus.log"
