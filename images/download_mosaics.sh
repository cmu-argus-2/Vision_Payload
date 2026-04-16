#!/bin/bash
# download_mosaics.sh
# Script to download images with exact pixel dimensions

# Configuration
MGRS_REGION="10S"              # MGRS grid region
START_DATE="2014"        # Start date
END_DATE="2025"          # End date
NUM_IMAGES=1000                  # Number of images
SENSOR="l8"                     # Sensor: l8, l9, or s2
GSD=175.0                       # Ground Sample Distance in meters
WIDTH_PIXELS=4608               # Image width in pixels
HEIGHT_PIXELS=2592              # Image height in pixels
OUTPUT_DIR="10S_images"   # Output directory on Google Drive

# Run the downloader with pixel dimensions
python eedl.py \
    -g "$MGRS_REGION" \
    -i "$START_DATE" \
    -f "$END_DATE" \
    -m "$NUM_IMAGES" \
    -se "$SENSOR" \
    -s "$GSD" \
    -wp "$WIDTH_PIXELS" \
    -hp "$HEIGHT_PIXELS" \
    -o "$OUTPUT_DIR" \
    -cm True \
    -gd True \
    -ba B4 B3 B2

echo ""
echo "Tasks submitted to Google Earth Engine!"
echo "Check status at: https://code.earthengine.google.com/tasks"
