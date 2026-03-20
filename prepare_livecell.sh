#!/bin/bash
#===============================================================================
# Download LIVECell dataset for cell segmentation fine-tuning
#
# Downloads microscopy images (~1.3 GB) and COCO annotations (~0.9 GB)
# from the LIVECell S3 bucket.
#
# Usage:
#   ./prepare_livecell.sh [--data-dir=/path/to/livecell]
#
# Default: /nrs/ml_datasets/livecell
# Total: ~2 GB after extraction
#===============================================================================

DATA_DIR="/nrs/ml_datasets/livecell"

for arg in "$@"; do
    case $arg in
        --data-dir=*) DATA_DIR="${arg#*=}" ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "================================================================"
echo "LIVECell Dataset Download"
echo "================================================================"
echo "Target: $DATA_DIR"
echo ""

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE_URL="http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"

# Download images
if [ -d "images/livecell_train_val_images" ]; then
    echo "Images already present — skipping download"
else
    echo "Downloading images (1.3 GB)..."
    wget -q --show-progress "$BASE_URL/images.zip" -O images.zip
    echo "Extracting..."
    unzip -q images.zip
    rm images.zip
    echo "Images extracted"
fi

# Download annotations
ANN_URL="$BASE_URL/annotations/LIVECell"
for split in train val test; do
    FILE="livecell_coco_${split}.json"
    if [ -f "$FILE" ]; then
        echo "Annotation $FILE already present — skipping"
    else
        echo "Downloading $FILE..."
        wget -q --show-progress "$ANN_URL/$FILE" -O "$FILE"
    fi
done

echo ""
echo "================================================================"
echo "LIVECell dataset ready at $DATA_DIR"
echo ""
TRAIN=$(ls images/livecell_train_val_images/ 2>/dev/null | wc -l)
TEST=$(ls images/livecell_test_images/ 2>/dev/null | wc -l)
echo "  Train/val images: $TRAIN"
echo "  Test images:      $TEST"
echo "  Annotations:      $(ls livecell_coco_*.json 2>/dev/null | wc -l) files"
echo "  Total size:       $(du -sh . | cut -f1)"
echo "================================================================"
