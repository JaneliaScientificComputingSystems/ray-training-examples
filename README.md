# Ray Distributed Training Examples — Janelia HPC

A practical guide and set of example scripts for running distributed PyTorch training with Ray on Janelia's GPU cluster. Covers all GPU tiers: H200 and H100 nodes with InfiniBand fabric, and L4/A100 nodes with Ethernet. The submission script auto-configures NCCL settings based on the queue you select.

---

## GPU Cluster Overview

| Queue | GPU | GPUs/node | Nodes | Total GPUs | Network | CPUs/node |
|---|---|---|---|---|---|---|
| `gpu_h200_parallel` | H200 | 8 | 8 | 64 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_h100_parallel` | H100 | 8 | 8 | 64 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_l4_parallel` | L4 | 8 | many | varies | Ethernet | 64 |
| `gpu_a100_parallel` | A100 | 4 | varies | varies | Ethernet | 48 |

**When to use H100/H200 (InfiniBand):** Large models (1B+ parameters), long runs (50+ epochs), or jobs scaling beyond 2 nodes. InfiniBand provides ~400 Gb/s per GPU rail with GPUDirect RDMA — GPU memory transfers to the NIC without CPU involvement. Ethernet NCCL uses TCP through the CPU with shared bandwidth.

**Rule of thumb:** Use H100/H200 for heavy multi-node jobs. L4 nodes are well-suited for smaller models, development runs, and single-node work.

### InfiniBand topology (H100 and H200 — identical configuration)

Each node has 8 IB HCAs, one per GPU, all at PIX distance (same PCIe switch). The fabric is NDR non-blocking — 64 simultaneous GPU-to-GPU transfers at ~400 Gb/s per rail with no queuing.

| GPU | IB HCA | PCIe relationship | NUMA |
|---|---|---|---|
| GPU 0 | mlx5_0 | PIX | 0 |
| GPU 1 | mlx5_1 | PIX | 0 |
| GPU 2 | mlx5_2 | PIX | 0 |
| GPU 3 | mlx5_3 | PIX | 0 |
| GPU 4 | mlx5_4 | PIX | 1 |
| GPU 5 | mlx5_6 | PIX | 1 |
| GPU 6 | mlx5_7 | PIX | 1 |
| GPU 7 | mlx5_8 | PIX | 1 |

> **Important:** `mlx5_5` is the host management **Ethernet** interface — not InfiniBand. It is intentionally excluded from `NCCL_IB_HCA`. Including it causes NCCL to attempt IB operations over Ethernet, resulting in job failure or severe performance regression.

To verify topology on any node:

```bash
nvidia-smi topo -m
ibv_devinfo | grep hca_id
for dev in mlx5_0 mlx5_1 mlx5_2 mlx5_3 mlx5_4 mlx5_5 mlx5_6 mlx5_7 mlx5_8; do
    echo -n "$dev: "; ibv_devinfo -d $dev | grep link_layer
done
```

---

## Prerequisites

- LSF cluster access with GPU queue permissions
- Python 3.8+
- Shared storage for datasets and model checkpoints (`/nrs/` or home directory)
- For H100/H200 jobs: IB drivers present on compute nodes (`ibv_devinfo` returns device list)

---

## Installation

One-time setup:

```bash
# Create and activate virtual environment
python -m venv ~/ray_env
source ~/ray_env/bin/activate

# Install Ray with training components
pip install "ray[train]" torch torchvision torchaudio

# Verify
python -c "import ray, torch, torchvision; print('Installation successful')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

For GPT-2 training, also install:

```bash
pip install transformers datasets tiktoken
```

---

## Files in This Repository

| File | Description |
|---|---|
| `submit_ray_job.sh` | Universal LSF job submission script — auto-configures NCCL per queue |
| `test_ray_cluster.py` | Validates cluster setup, GPU access, and network backend |
| `cifar10_distributed_training.py` | ResNet-18 on CIFAR-10 — multi-node DDP training |
| `image_classifier.py` | CIFAR-10 inference with trained ResNet-18 checkpoints |
| `gpt2_distributed_training.py` | GPT-2 small (117M) on OpenWebText — DDP and FSDP modes |
| `gpt2_eval.py` | Perplexity evaluation on OpenWebText val set |
| `gpt2_generate.py` | Text generation — interactive, batch, and single-prompt modes |

---

## Submission Script

Make the script executable once:

```bash
chmod +x submit_ray_job.sh
```

The script automatically selects the correct NCCL configuration based on the queue:

- **H100/H200 queues** — InfiniBand enabled, GPUDirect RDMA, `mlx5_5` excluded
- **L4/A100 queues** — Ethernet NCCL

### Usage

```
./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]

Required:
  <num_nodes>          Number of nodes
  --script=FILE        Python script to run

Optional:
  --queue=QUEUE        LSF queue (default: gpu_l4_parallel)
  --job-name=NAME      Job name (default: ray_job)
  --walltime=TIME      Walltime in hours (default: 4:00 for <=2 nodes, 8:00 otherwise)
  --venv=PATH          Python venv path

Script arguments: pass after '--'
  Example: ./submit_ray_job.sh 2 --script=train.py -- --epochs=50 --batch-size=128
```

---

## Step 1: Test Your Cluster

Always run the cluster test before submitting expensive training jobs.

```bash
# L4 nodes — single node
./submit_ray_job.sh 1 --queue=gpu_l4_parallel --venv=~/ray_env \
    --script=test_ray_cluster.py

# H100 nodes — two nodes (recommended before any multi-node training run)
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --script=test_ray_cluster.py

# H200 nodes — two nodes
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=test_ray_cluster.py
```

**Expected output (H200, 2 nodes):**

```
======================================================================
RAY CLUSTER TEST
======================================================================
Connected to Ray cluster
  CPUs:   192.0
  GPUs:   16.0
  Memory: 1480.3 GB
  NCCL backend: InfiniBand
  IB HCAs: mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8

Testing 16 GPUs...
  GPU 0: NVIDIA H200 — OK
  ...

Cluster OK — 16.0 GPUs | InfiniBand NCCL
======================================================================
```

For IB queues, also confirm NCCL selected the IB path (not TCP fallback):

```bash
grep "NET/IB\|NET/Socket" ray_job_*.out | head -5
# Good: NCCL INFO Channel 00/08 : 0[mlx5_0] -> 8[mlx5_0] [send] via NET/IB/0
# Bad:  NCCL INFO ... via NET/Socket   <- TCP fallback, check NCCL_IB_HCA names
```

---

## Example 1: CIFAR-10 Distributed Training

Trains ResNet-18 on CIFAR-10. Works on all queues — DataLoader settings adapt based on available CPUs.

### Dataset setup (once)

```bash
mkdir -p ~/datasets/cifar10 && cd ~/datasets/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz

# Verify
python -c "
import torchvision, os
ds = torchvision.datasets.CIFAR10(
    root=os.path.expanduser('~/datasets/cifar10'), train=True, download=False)
print(f'CIFAR-10 ready: {len(ds)} training samples')
"
```

### Run training

**H200 nodes — InfiniBand:**

```bash
# 2 nodes, 16 GPUs
./submit_ray_job.sh 2 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=cifar10 \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=256 --save-models

# 8 nodes, 64 GPUs (full cluster)
./submit_ray_job.sh 8 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=cifar10 \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --epochs=50 --batch-size=512 --save-models
```

**H100 nodes — InfiniBand:**

```bash
./submit_ray_job.sh 2 \
    --queue=gpu_h100_parallel \
    --venv=~/ray_env \
    --job-name=cifar10 \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=256 --save-models
```

**L4 nodes — Ethernet:**

```bash
# 2 nodes, 16 GPUs
./submit_ray_job.sh 2 \
    --queue=gpu_l4_parallel \
    --venv=~/ray_env \
    --job-name=cifar10 \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=20 --batch-size=128 --save-models

# Single node, 8 GPUs
./submit_ray_job.sh 1 \
    --queue=gpu_l4_parallel \
    --venv=~/ray_env \
    --job-name=cifar10 \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=8 --num-nodes=1 --epochs=20 --save-models
```

### Resume from checkpoint

```bash
# Find last checkpoint
ls -lt ./models/cifar10_resnet18_best_*.pth | head -1

# Resume
./submit_ray_job.sh 2 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=cifar10_resume \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=256 --save-models \
    --resume=./models/cifar10_resnet18_best_YYYYMMDD_HHMMSS.pth
```

### Run inference

```bash
# Find best model
ls -lt ./models/cifar10_resnet18_best_*.pth | head -1

# Test on CIFAR-10 samples
python image_classifier.py \
    --model ./models/cifar10_resnet18_best_YYYYMMDD_HHMMSS.pth \
    --test

# Classify a single image
python image_classifier.py \
    --model ./models/cifar10_resnet18_best_YYYYMMDD_HHMMSS.pth \
    --image ~/test_images/cat.jpg
```

---

## Example 2: GPT-2 Distributed Training

Trains GPT-2 small (117M parameters) on OpenWebText. Runs 3–6 hours at full cluster scale — long enough to validate IB throughput, generate real scaling numbers, and demonstrate FSDP vs DDP on a model that actually benefits from parameter sharding.

**Why this benchmark matters:** GPT-2 AllReduce payloads are ~1.7 GB per step (all gradients across 117M params). At 64 GPUs that gradient sync happens every forward/backward pass — your IB fabric is doing continuous heavy lifting, not occasional bursts. FSDP adds AllGather on every forward pass too, pushing inter-node bandwidth even harder.

### Dataset setup (once)

OpenWebText is ~40 GB uncompressed. Download and tokenize once to shared storage — this takes 30–60 minutes.

```bash
mkdir -p ~/datasets/openwebtext
cd ~/datasets/openwebtext

python3 - << 'EOF'
from datasets import load_dataset
import tiktoken
import numpy as np
import os

print("Downloading OpenWebText (~40GB, this takes a while)...")
dataset = load_dataset("openwebtext", num_proc=8)

enc = tiktoken.get_encoding("gpt2")
EOT = enc._special_tokens["<|endoftext|>"]

def tokenise(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(EOT)
    return {"ids": ids, "len": len(ids)}

print("Tokenising...")
tokenised = dataset.map(tokenise, remove_columns=["text"],
                        desc="tokenising", num_proc=8)

for split, dset in tokenised["train"].train_test_split(
        test_size=0.0005, seed=42).items():
    name = "train" if split == "train" else "val"
    arr_len = sum(dset["len"])
    out = np.memmap(f"{name}.bin", dtype=np.uint16, mode="w+", shape=(arr_len,))
    idx = 0
    for batch in dset.iter(batch_size=1024):
        chunk = np.concatenate(batch["ids"]).astype(np.uint16)
        out[idx:idx+len(chunk)] = chunk
        idx += len(chunk)
    out.flush()
    print(f"{name}.bin written — {arr_len:,} tokens ({arr_len*2/1e9:.1f} GB)")

print("Dataset ready.")
EOF
```

Verify:

```bash
ls -lh ~/datasets/openwebtext/
# train.bin  ~9.0 GB   (~4.5B tokens)
# val.bin    ~4.5 MB
python3 -c "
import numpy as np
t = np.memmap('train.bin', dtype=np.uint16, mode='r')
print(f'Train tokens: {len(t):,}')
"
```

### Run training

**DDP mode — H200 (recommended first run):**

```bash
# Validation run — 2 nodes, 16 GPUs, 2000 iters (~45 min)
./submit_ray_job.sh 2 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_ddp \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models

# Full run — 8 nodes, 64 GPUs, 50K iters (~6h)
./submit_ray_job.sh 8 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_ddp_full \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --mode=ddp \
    --max-iters=50000 --batch-size=8 --grad-accum=4 --save-models
```

**FSDP mode — H200:**

FSDP shards model parameters across all GPUs in a node group. Use when the model does not fit in single GPU memory, or to compare AllGather overhead vs DDP AllReduce on your IB fabric.

```bash
./submit_ray_job.sh 2 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_fsdp \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=fsdp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models
```

**H100 and L4 queues:**

```bash
# H100 — identical to H200 commands, swap queue name
./submit_ray_job.sh 2 \
    --queue=gpu_h100_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_ddp \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models

# L4 — reduce batch size; Ethernet limits throughput on multi-node
./submit_ray_job.sh 2 \
    --queue=gpu_l4_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_ddp \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=4 --grad-accum=8 --save-models
```

### Resume from checkpoint

```bash
ls -lt ./models/gpt2_ddp_best_*.pth | head -1

./submit_ray_job.sh 8 \
    --queue=gpu_h200_parallel \
    --venv=~/ray_env \
    --job-name=gpt2_resume \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --mode=ddp \
    --max-iters=50000 --batch-size=8 --grad-accum=4 --save-models \
    --resume=./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth
```

### Expected throughput and IB validation

Use these numbers to confirm your IB fabric and GPUDirect are working correctly. Significantly lower throughput suggests NCCL fell back to TCP — check `grep "NET/IB" gpt2_ddp_*.out`.

| Queue | Mode | GPUs | Tokens/sec | Notes |
|---|---|---|---|---|
| `gpu_h200_parallel` | DDP | 8 | ~280K | single node baseline |
| `gpu_h200_parallel` | DDP | 16 | ~540K | ~1.9× — IB scaling |
| `gpu_h200_parallel` | DDP | 64 | ~2.0M | ~7.1× — full cluster |
| `gpu_h200_parallel` | FSDP | 16 | ~480K | slightly lower — AllGather overhead |
| `gpu_h100_parallel` | DDP | 16 | ~400K | H100 baseline |
| `gpu_l4_parallel` | DDP | 16 | ~90K | Ethernet — bandwidth limited at 2+ nodes |

> **DDP vs FSDP for GPT-2 small:** At 117M parameters both perform similarly — the model fits in single GPU memory. FSDP becomes the right choice at ~1B+ params where per-GPU memory is the constraint. Run both modes to establish baseline numbers; the FSDP overhead on IB is a useful data point when planning larger model runs.

### Batch size reference

| GPU | Recommended batch size | Seq len | Notes |
|---|---|---|---|
| H200 (141 GB) | 12–16 | 1024 | headroom for larger seq_len |
| H100 (80 GB) | 8–12 | 1024 | |
| L4 (24 GB) | 4–6 | 1024 | reduce seq_len to 512 if OOM |
| A100 (40/80 GB) | 6–10 | 1024 | |

### Evaluate perplexity

```bash
# Quick eval — 200 batches (~2 min on single GPU)
python gpt2_eval.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --num-batches 200

# Full val set eval
python gpt2_eval.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth

# Compare DDP vs FSDP checkpoint quality
python gpt2_eval.py --model ./models/gpt2_ddp_best_*.pth
python gpt2_eval.py --model ./models/gpt2_fsdp_best_*.pth
```

Expected output:

```
Loaded: ./models/gpt2_ddp_best_20251205_143022.pth
  Training mode:  DDP
  Saved at iter:  5000
  Saved val loss: 3.4200  (ppl 30.6)

Evaluating on val set (200 batches)...
  batch   50 | running ppl 30.82
  ...

==================================================
Val loss:    3.4210
Perplexity:  30.61
Bits/char:   1.234  (estimate, ~4 chars/token)
Tokens eval: 1,638,400
==================================================

Reference perplexities (OpenWebText val):
  GPT-2 small  (117M) trained to convergence: ~29 ppl
  GPT-2 medium (345M) trained to convergence: ~24 ppl
  Random baseline (50K vocab):               ~50000 ppl
```

### Generate text

```bash
# Single prompt
python gpt2_generate.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --prompt "The human brain contains approximately" \
    --max-tokens 150

# Multiple samples (explore diversity)
python gpt2_generate.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --prompt "Scientists recently discovered" \
    --num-samples 3 --temperature 0.9 --max-tokens 100

# Greedy decoding (deterministic)
python gpt2_generate.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --prompt "In summary, the results show" \
    --temperature 0 --max-tokens 100

# Interactive REPL
python gpt2_generate.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --interactive

# Batch prompts from a file
cat > prompts.txt << 'EOF'
The experiment demonstrated that
Neural networks trained on large datasets
The most important finding was
EOF
python gpt2_generate.py \
    --model ./models/gpt2_ddp_best_YYYYMMDD_HHMMSS.pth \
    --prompts-file prompts.txt --max-tokens 120
```

| Flag | Default | Notes |
|---|---|---|
| `--temperature` | 0.8 | Lower = more focused, higher = more creative. 0 = greedy |
| `--top-k` | 200 | Restricts sampling to top K tokens. 0 to disable |
| `--num-samples` | 1 | Independent samples per prompt |
| `--max-tokens` | 200 | New tokens to generate beyond the prompt |
| `--interactive` | off | REPL mode — type prompts at the terminal |

---

## Monitoring and Troubleshooting

### Monitor a running job

```bash
bjobs
tail -f ray_job_*.out
tail -f ray_job_*.err

# IB queues — confirm NCCL selected InfiniBand
grep "NET/IB\|NET/Socket" ray_job_*.out | head -5
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NET/Socket` in output on H100/H200 | NCCL fell back to TCP | Confirm `NCCL_IB_DISABLE=0` and HCA names with `ibv_devinfo` |
| Job hangs at first AllReduce on IB node | `mlx5_5` in `NCCL_IB_HCA` | Remove `mlx5_5` — it is Ethernet |
| `ibv_devinfo` fails on compute node | IB kernel modules not loaded | Run `lsmod | grep ib` and contact SCS |
| IB bandwidth low in `ib_send_bw` | Cable, SFP, or switch port issue | Check `ibstat` for port state; escalate to SCS if not Active |
| Out of memory | Batch size too large | Reduce `--batch-size`; start at 64–128 per GPU for CIFAR-10 |
| Ray connection timeout | Head node slow to start | Increase `sleep 20` in the submission script to `sleep 30` |
| Wrong GPU count | Queue mismatch | A100 = 4 GPUs/node, all others = 8 |
| L4 job slow on multi-node | Expected — Ethernet bandwidth shared | Reduce nodes, increase batch size to amortize sync cost |

### Verify IB bandwidth between two nodes

```bash
# On node A:
ib_send_bw -d mlx5_0 -i 1 -F --report_gbits

# On node B:
ib_send_bw -d mlx5_0 -i 1 -F --report_gbits <nodeA_hostname>

# Expected: ~380-400 Gb/s per rail (NDR IB)
```

---

## Questions and Support

For cluster access, queue permissions, or IB hardware issues, contact the Scientific Computing Support team.

For issues with these example scripts, open an issue at [JaneliaScientificComputingSystems/ray-training-examples](https://github.com/JaneliaScientificComputingSystems/ray-training-examples).
