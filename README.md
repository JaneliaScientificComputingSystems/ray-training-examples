# Ray Distributed Training Examples

Distributed PyTorch training with Ray on Janelia's GPU cluster. The submission script auto-configures NCCL for InfiniBand (H200/H100) or Ethernet (L4/A100) based on the queue you select.

Three ready-to-run examples that progressively exercise the cluster:

| Example | Model | Dataset | Purpose |
|---------|-------|---------|---------|
| CIFAR-10 | ResNet-18 (11M) | 60K images, 170 MB | Quick smoke test — minutes on any queue |
| GPT-2 | GPT-2 small (117M) | OpenWebText, 9 GB | Language model with sustained IB load (~1.7 GB allreduce/step) |
| ImageNet | ResNet-50 (25.6M) | ImageNet-1K, 138 GB | Industry-standard benchmark for cluster validation |

---

## GPU Queues

| Queue | GPU | GPUs/node | Network | CPUs/node |
|-------|-----|-----------|---------|-----------|
| `gpu_h200_parallel` | H200 | 8 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_h100_parallel` | H100 | 8 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_l4_parallel` | L4 | 8 | Ethernet | 64 |
| `gpu_a100_parallel` | A100 | 4 | Ethernet | 48 |

Use H200/H100 for multi-node jobs and large models. L4 nodes work well for single-node training and development.

---

## Installation

```bash
git clone https://github.com/JaneliaScientificComputingSystems/ray-training-examples.git
cd ray-training-examples
chmod +x submit_ray_job.sh prepare_openwebtext.sh prepare_imagenet.sh

python -m venv ~/ray_env
source ~/ray_env/bin/activate
pip install "ray[train]" torch torchvision torchaudio

# For GPT-2 and ImageNet examples
pip install datasets tiktoken
```

---

## Files

| File | Description |
|------|-------------|
| `submit_ray_job.sh` | LSF job submission — auto-configures NCCL per queue |
| `run_inference.sh` | LSF job submission for inference (single GPU) |
| `cifar10_distributed_training.py` | ResNet-18 on CIFAR-10 — multi-node DDP |
| `prepare_openwebtext.sh` | Downloads and tokenizes OpenWebText (already done — for reference) |
| `gpt2_distributed_training.py` | GPT-2 small (117M) — DDP and FSDP modes |
| `prepare_imagenet.sh` | Downloads ImageNet-1K (already done — for reference) |
| `imagenet_distributed_training.py` | ResNet-50 on ImageNet-1K — distributed benchmark |
| `image_classifier.py` | CIFAR-10 inference with trained checkpoints |
| `gpt2_eval.py` | GPT-2 perplexity evaluation on val set |
| `gpt2_generate.py` | GPT-2 text generation from checkpoints |
| `imagenet_classifier.py` | ImageNet inference with trained checkpoints |

---

## Submission Script

```
./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]

Options:
  --queue=QUEUE        GPU queue (default: gpu_l4_parallel)
  --venv=PATH          Python venv path
  --job-name=NAME      Job name (default: ray_job)
  --walltime=TIME      Walltime (default: 4:00 for <=2 nodes, 8:00 otherwise)

Script args go after '--':
  ./submit_ray_job.sh 2 --script=train.py -- --epochs=50
```

---

## Example 1: CIFAR-10

### About the dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is 60,000 color images (32x32 pixels) in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. 50,000 training images and 10,000 test images. At 170 MB it's small enough to load instantly on any node.

### What the training does

Trains a ResNet-18 (11M parameters) image classifier using Distributed Data Parallel (DDP). Each GPU processes a different mini-batch, gradients are synchronized via allreduce after each step. Standard data augmentation (random crop, horizontal flip) is applied. The model learns to classify the 10 object categories.

### What it produces

- Training logs with per-epoch accuracy and loss
- Model checkpoints (`.pth` files) saved to `../models/` when `--save-models` is used
- Best checkpoint based on test accuracy (~89-91% after 50 epochs)
- Checkpoints can be used with `image_classifier.py` for inference on new images

### Data

CIFAR-10 is pre-installed at `/nrs/ml_datasets/cifar10` (shared NFS). All scripts use this by default — just run them, no data setup needed.

<details><summary>Preparing your own copy (optional — only if you need a separate location)</summary>

```bash
mkdir -p /your/path/cifar10 && cd /your/path/cifar10
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz && rm cifar-10-python.tar.gz
```

</details>

### Train

**Quick validation (~2 min on 16 GPUs):** 10 epochs is enough to confirm multi-node training works and see loss converge. Expect ~80% accuracy.

**Full training (~5 min on 16 GPUs):** 50 epochs reaches ~89-91% accuracy — diminishing returns beyond that on CIFAR-10.

```bash
# Validation run — 10 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=10 --batch-size=256 --save-models

# Full run — 50 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=256 --save-models

# L4 — 1 node, 8 GPUs
./submit_ray_job.sh 1 --queue=gpu_l4_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=8 --num-nodes=1 --epochs=20 --save-models
```

### Inference

Inference scripts require a GPU — run them as LSF jobs, not on the login node:

```bash
# Test on CIFAR-10 samples
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --test

# Classify a single image
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --image photo.jpg
```

---

## Example 2: GPT-2

### About the dataset

[OpenWebText](https://huggingface.co/datasets/openwebtext) is a reproduction of the WebText corpus used to train the original GPT-2. It contains ~8 million web pages scraped from URLs shared on Reddit with at least 3 karma. The text is tokenized with GPT-2's BPE tokenizer (50,257 token vocabulary) into ~4.5 billion tokens. A 0.05% held-out validation split is used for perplexity evaluation.

### What the training does

Trains GPT-2 small (117M parameters, 12 layers, 768 hidden dim, 12 attention heads) to predict the next token in a sequence. Uses DDP or FSDP for distributed training with gradient accumulation (effective batch = batch_size × grad_accum × num_gpus sequences). Mixed precision (bfloat16 on H200/H100) with cosine LR schedule and warmup. Each training step produces ~1.7 GB of gradient data that must be synchronized across all GPUs via allreduce — this creates sustained IB traffic.

### What it produces

- Training logs with loss, learning rate, and tokens/sec throughput
- Periodic validation loss (perplexity) evaluation
- Model checkpoints (`.pth` files) saved to `../models/`
- Trained models can be evaluated with `gpt2_eval.py` or used for text generation with `gpt2_generate.py`

### Data

Tokenized OpenWebText is pre-installed at `/nrs/ml_datasets/openwebtext` (shared NFS). All scripts use this by default — just run them, no data setup needed.

<details><summary>Preparing your own copy (optional — only if you need a separate location)</summary>

```bash
pip install datasets tiktoken
./prepare_openwebtext.sh --data-dir=/your/path
```

</details>

### Train

GPT-2 uses iteration count, not epochs. Each iteration processes `batch_size × grad_accum × num_gpus` sequences (1024 tokens each). With the defaults below, effective batch = 1,048,576 tokens/iter.

**Quick validation (~45 min on 16 GPUs):** 2000 iters processes ~2B tokens — enough to verify the pipeline, see loss drop, and confirm IB throughput.

**Full training (~6h on 16 GPUs):** 50K iters processes ~50B tokens (~5 passes over OpenWebText). Expect perplexity ~29 at convergence.

```bash
# Validation run — 2000 iters
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models

# Full run — 50K iters
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=50000 --batch-size=8 --grad-accum=8 --save-models

# FSDP mode (for models that don't fit in single GPU memory)
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=fsdp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models
```

### Evaluate & Generate

```bash
# Perplexity evaluation
./run_inference.sh --venv=~/ray_env --script=gpt2_eval.py -- \
    --model ../models/gpt2_ddp_best.pth --num-batches 200

# Text generation
./run_inference.sh --venv=~/ray_env --script=gpt2_generate.py -- \
    --model ../models/gpt2_ddp_best.pth --prompt "The brain"

# Interactive mode
./run_inference.sh --venv=~/ray_env --script=gpt2_generate.py -- \
    --model ../models/gpt2_ddp_best.pth --interactive
```

---

## Example 3: ResNet-50 on ImageNet-1K

### About the dataset

[ImageNet-1K (ILSVRC2012)](https://huggingface.co/datasets/ILSVRC/imagenet-1k) is the standard benchmark for image classification. 1,281,167 training images and 50,000 validation images across 1,000 object categories ranging from animals (tench, goldfinch, tree frog) to objects (laptop, sunglasses, volcano) to scenes (lakeside, valley). Images are variable-size JPEGs, resized to 224x224 during training. This is the dataset used in MLPerf benchmarks and NVIDIA DGX cluster validation.

### What the training does

Trains ResNet-50 (25.6M parameters) using DDP with large-batch SGD (momentum 0.9, weight decay 1e-4). Learning rate is linearly scaled by effective batch size (lr = base_lr × effective_batch / 256) with linear warmup and cosine decay. Mixed precision (bfloat16). Data is streamed via Ray Data from parquet shards — no DataLoader subprocesses or memmapping. Each training step produces ~100 MB of gradient data synchronized via allreduce, creating sustained IB traffic at scale.

### What it produces

- Training logs with per-epoch top-1 accuracy, loss, learning rate, and images/sec throughput
- Periodic validation with top-1 and top-5 accuracy
- Model checkpoints (`.pth` files) saved to `../models/`
- Target: ~76%+ top-1 accuracy after 90 epochs (~3h on 16x H200)

### Data

ImageNet-1K is pre-installed at `/nrs/ml_datasets/imagenet` (shared NFS, ~138 GB). All scripts use this by default — just run them, no data setup needed.

<details><summary>Preparing your own copy (optional — only if you need a separate location)</summary>

1. Create a [HuggingFace](https://huggingface.co) account
2. Accept the license at [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
3. Create a token at [Settings > Tokens](https://huggingface.co/settings/tokens)

```bash
export HF_TOKEN="hf_..."
./prepare_imagenet.sh --data-dir=/your/path
```

</details>

### Train

**Quick validation (~20 min on 16 GPUs):** 3 epochs is enough to confirm the data pipeline works, see loss drop, and measure throughput. Expect ~25-30% top-1.

**Full training (~3h on 16 GPUs):** 90 epochs reaches ~76%+ top-1 accuracy — the standard ResNet-50 target used in MLPerf benchmarks.

```bash
# Validation run — 3 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=3 --batch-size=128 --save-models

# Full run — 90 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=90 --batch-size=128 --save-models

# 8 nodes, 64 GPUs
./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=resnet50 \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --epochs=90 --batch-size=64 --save-models
```

### Inference

```bash
./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --test

./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --image photo.jpg
```

### Batch size reference

LR is automatically scaled: `lr = base_lr * effective_batch / 256`.

| GPU | Per-GPU batch | Notes |
|-----|--------------|-------|
| H200 (141 GB) | 128–256 | |
| H100 (80 GB) | 128 | |
| L4 (24 GB) | 32–64 | |

---

### Resume from checkpoint

All scripts support `--resume`:

```bash
# Resume CIFAR-10
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=cifar10_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --save-models \
    --resume=../models/cifar10_resnet18_latest.pth

# Resume ImageNet
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=90 --batch-size=128 --save-models \
    --resume=../models/resnet50_imagenet_latest.pth

# Resume GPT-2
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp --max-iters=5000 --save-models \
    --resume=../models/gpt2_ddp_latest.pth
```

Checkpoints: `--save-models` saves `<model>_latest.pth` (every epoch) and `<model>_best.pth` (on improvement) to `../models/`.

---

## Monitoring

Job output goes to `../output/`:

```bash
bjobs                              # list jobs
tail -f ../output/*_<JOBID>.out    # training output
tail -f ../output/*_<JOBID>.err    # errors
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` |
| Ray timeout | Resubmit — transient scheduling issue |
| Wrong GPU count | A100 = 4/node, all others = 8/node |
| L4 slow multi-node | Expected — use fewer nodes or larger batch |
| NCCL errors or hangs | Contact SCS |
