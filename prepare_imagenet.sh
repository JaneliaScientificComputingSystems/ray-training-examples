#!/bin/bash
#===============================================================================
# Download ImageNet-1K (ILSVRC2012) from HuggingFace
#
# Downloads ~138 GB to /nrs/scicompsys/Goran/imagenet
# Requires HuggingFace token with access to ILSVRC/imagenet-1k
#
# Usage:
#   ./prepare_imagenet.sh [--venv=PATH] [--data-dir=PATH]
#
# Defaults:
#   venv:     ~/ray_env
#   data-dir: /nrs/scicompsys/Goran/imagenet
#===============================================================================

VENV_PATH="${HOME}/ray_env"
DATA_DIR="/nrs/scicompsys/Goran/imagenet"

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "ImageNet-1K (ILSVRC2012) Download"
echo "================================================================"
echo "Venv:     $VENV_PATH"
echo "Data dir: $DATA_DIR"
echo "Size:     ~138 GB"
echo "================================================================"

cat << EOF | bsub
#!/bin/bash
#BSUB -J imagenet_download
#BSUB -n 8
#BSUB -o ../output/imagenet_download_%J.out
#BSUB -e ../output/imagenet_download_%J.err
#BSUB -W 24:00

source ${VENV_PATH}/bin/activate

mkdir -p ${DATA_DIR}

echo "Downloading ImageNet-1K (ILSVRC2012) from HuggingFace..."
echo "Target: ${DATA_DIR}"
echo ""

python3 - << 'PYEOF'
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

from huggingface_hub import snapshot_download

data_dir = "${DATA_DIR}"
print(f"Downloading to: {data_dir}")
print("This will take a while (~138 GB)...")

snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir=data_dir,
)

print()
print("Download complete.")
for f in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, f)
    if os.path.isfile(path):
        size = os.path.getsize(path) / 1e9
        print(f"  {f}: {size:.1f} GB")
PYEOF

echo ""
echo "ImageNet download complete. Building Arrow cache..."

python3 - << 'PYEOF2'
import os

data_dir = "${DATA_DIR}"
hf_cache = os.path.join(data_dir, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = hf_cache
os.environ["HF_HOME"] = hf_cache

from datasets import load_dataset

parquet_dir = os.path.join(data_dir, "data")
if not os.path.isdir(parquet_dir):
    parquet_dir = data_dir

print("Loading parquet shards into Arrow cache (single process, ~6 min)...")
dataset = load_dataset(
    "parquet",
    data_files={
        "train": os.path.join(parquet_dir, "train-*.parquet"),
        "validation": os.path.join(parquet_dir, "validation-*.parquet"),
    },
    cache_dir=hf_cache,
)
print(f"  Train: {len(dataset['train']):,} images")
print(f"  Val:   {len(dataset['validation']):,} images")
print(f"  Cache: {hf_cache}")
print("Arrow cache built. Training runs will reuse it instantly.")
PYEOF2

echo ""
echo "ImageNet preparation complete."
ls -lh ${DATA_DIR}/
EOF
