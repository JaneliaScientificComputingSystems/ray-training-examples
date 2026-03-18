#!/bin/bash
#===============================================================================
# Download and tokenize OpenWebText for GPT-2 training
#
# Downloads from HuggingFace, tokenizes with GPT-2 BPE, writes parquet shards.
# Takes 30-60 minutes as a CPU job.
#
# Usage:
#   ./prepare_openwebtext.sh [--venv=PATH] [--data-dir=PATH]
#
# Defaults:
#   venv:     ~/ray_env
#   data-dir: /nrs/scicompsys/Goran/openwebtext
#
# Output:
#   <data-dir>/train/*.parquet  (~9 GB, ~4.5B tokens)
#   <data-dir>/val/*.parquet    (~4.5 MB)
#===============================================================================

VENV_PATH="${HOME}/ray_env"
DATA_DIR="/nrs/scicompsys/Goran/openwebtext"

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv=*)     VENV_PATH="${1#*=}"; shift ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo "OpenWebText Dataset Preparation"
echo "================================================================"
echo "Venv:     $VENV_PATH"
echo "Data dir: $DATA_DIR"
echo "================================================================"

cat << EOF | bsub
#!/bin/bash
#BSUB -J openwebtext_prep
#BSUB -n 8
#BSUB -o ../output/openwebtext_prep_%J.out
#BSUB -e ../output/openwebtext_prep_%J.err
#BSUB -W 4:00

source ${VENV_PATH}/bin/activate

mkdir -p ${DATA_DIR}/train ${DATA_DIR}/val

python3 - << 'PYEOF'
import os
import numpy as np
import tiktoken
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "${DATA_DIR}"
BLOCK_SIZE = 1024

print("Step 1/4: Downloading OpenWebText from HuggingFace...")
from datasets import load_dataset
dataset = load_dataset("openwebtext", trust_remote_code=True)
print(f"  Downloaded: {len(dataset['train'])} documents")

print("Step 2/4: Tokenizing with GPT-2 BPE tokenizer...")
enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

def tokenize_and_chunk(examples):
    """Tokenize text and chunk into fixed-length sequences."""
    all_tokens = []
    for text in examples["text"]:
        tokens = enc.encode_ordinary(text)
        tokens.append(EOT)
        all_tokens.extend(tokens)

    # Chunk into BLOCK_SIZE sequences
    num_chunks = len(all_tokens) // BLOCK_SIZE
    if num_chunks == 0:
        return {"input_ids": []}

    all_tokens = all_tokens[:num_chunks * BLOCK_SIZE]
    chunks = np.array(all_tokens, dtype=np.int64).reshape(num_chunks, BLOCK_SIZE)
    return {"input_ids": [row for row in chunks]}

tokenized = dataset.map(
    tokenize_and_chunk,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="tokenising",
    num_proc=8,
)

print("Step 3/4: Splitting train/val...")
split = tokenized["train"].train_test_split(test_size=0.0005, seed=42)
train_ds = split["train"]
val_ds = split["test"]

print(f"  Train: {len(train_ds):,} sequences ({len(train_ds) * BLOCK_SIZE:,} tokens)")
print(f"  Val:   {len(val_ds):,} sequences ({len(val_ds) * BLOCK_SIZE:,} tokens)")

print("Step 4/4: Writing parquet shards...")

# Write train shards (~100K sequences per shard)
SHARD_SIZE = 100000
train_dir = os.path.join(DATA_DIR, "train")
for i in range(0, len(train_ds), SHARD_SIZE):
    shard = train_ds.select(range(i, min(i + SHARD_SIZE, len(train_ds))))
    shard_num = i // SHARD_SIZE
    total_shards = (len(train_ds) + SHARD_SIZE - 1) // SHARD_SIZE
    path = os.path.join(train_dir, f"train-{shard_num:05d}-of-{total_shards:05d}.parquet")
    shard.to_parquet(path)
    if shard_num % 10 == 0:
        print(f"  train shard {shard_num}/{total_shards}")

# Write val as single shard
val_path = os.path.join(DATA_DIR, "val", "val-00000-of-00001.parquet")
val_ds.to_parquet(val_path)

train_size = sum(os.path.getsize(os.path.join(train_dir, f))
                 for f in os.listdir(train_dir)) / 1e9
print(f"\nDone.")
print(f"  Train: {train_dir} ({train_size:.1f} GB)")
print(f"  Val:   {val_path}")
print(f"  Sequences of {BLOCK_SIZE} tokens each")
PYEOF

echo "Dataset preparation complete."
ls -lh ${DATA_DIR}/train/ | tail -5
ls -lh ${DATA_DIR}/val/
EOF
