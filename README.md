# Ray Distributed Training Examples

Distributed PyTorch training with Ray on Janelia's GPU cluster. Uses InfiniBand NDR (400 Gb/s) with GPUDirect RDMA on H200/H100 nodes for high-performance multi-node training.

Five ready-to-run examples that progressively exercise the cluster:

| Example | Model | Dataset | Purpose |
|---------|-------|---------|---------|
| CIFAR-10 | ResNet-18 (11M) | 60K images, 170 MB | Quick smoke test — minutes on 2 nodes |
| GPT-2 | GPT-2 small–2.7B | OpenWebText, 9 GB | Language model — scales from quick tests to full IB stress tests |
| ImageNet ResNet | ResNet-50 (25.6M) | ImageNet-1K, 138 GB | Industry-standard CNN benchmark |
| ImageNet ViT | ViT base–huge (86M–632M) | ImageNet-1K, 138 GB | Vision Transformer — modern architecture, heavier IB load |
| Cell Segmentation | Swin-B (88M) | LIVECell, 2 GB | Fine-tuning a pretrained model for microscopy — transfer learning |

---

## GPU Queues

**Parallel queues** — whole nodes, multi-node training:

| Queue | GPU | GPUs/node | Network | CPUs/node |
|-------|-----|-----------|---------|-----------|
| `gpu_h200_parallel` | H200 | 8 | InfiniBand NDR 400 Gb/s | 96 |
| `gpu_h100_parallel` | H100 | 8 | InfiniBand NDR 400 Gb/s | 96 |

**Non-parallel queues** — single node, request 1-8 GPUs with `--num-gpus`:

| Queue | GPU | Max GPUs | Network |
|-------|-----|----------|---------|
| `gpu_h100` | H100 | 8 | InfiniBand |
| `gpu_h200` | H200 | 8 | InfiniBand |
| `gpu_l4` | L4 | 8 | Ethernet |
| `gpu_a100` | A100 | 4 | Ethernet |

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
| `gpt2_distributed_training.py` | GPT-2 (117M–2.7B) — DDP and FSDP modes |
| `prepare_imagenet.sh` | Downloads ImageNet-1K (already done — for reference) |
| `imagenet_distributed_training.py` | ResNet-50 on ImageNet-1K — distributed benchmark |
| `image_classifier.py` | CIFAR-10 inference with trained checkpoints |
| `gpt2_eval.py` | GPT-2 perplexity evaluation on val set |
| `gpt2_generate.py` | GPT-2 text generation from checkpoints |
| `vit_imagenet_distributed_training.py` | ViT (86M–632M) on ImageNet-1K — Vision Transformer DDP |
| `imagenet_classifier.py` | ImageNet inference (ResNet-50) with trained checkpoints |
| `vit_imagenet_classifier.py` | ImageNet inference (ViT) with trained checkpoints |
| `prepare_livecell.sh` | Downloads LIVECell dataset (already done — for reference) |
| `prepare_livecell_masks.py` | Precomputes semantic masks from COCO annotations (already done) |
| `livecell_finetune.py` | Swin-B fine-tuning on LIVECell — cell segmentation |
| `cell_segmenter.py` | Cell segmentation inference — produces colored masks |

---

## Submission Script

```
./submit_ray_job.sh <num_nodes> --script=SCRIPT [options] [-- script_args...]

Options:
  --queue=QUEUE        GPU queue (default: gpu_h100_parallel)
  --num-gpus=N         GPUs per node (non-parallel queues, default: 1)
  --num-cpus=N         CPUs to request (non-parallel queues, default: 12 per GPU)
  --venv=PATH          Python venv path
  --job-name=NAME      Job name (default: ray_job)
  --walltime=TIME      Walltime (default: 24h non-parallel, 4-8h parallel)

Script args go after '--':
  ./submit_ray_job.sh 2 --script=train.py -- --epochs=50
```

**Multi-node** (parallel queues — whole nodes, all GPUs):
```bash
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --script=train.py -- --num-gpus=16 --num-nodes=2
```

**Single-node development** (non-parallel queues — request specific GPUs):
```bash
./submit_ray_job.sh 1 --queue=gpu_h100 --num-gpus=2 --venv=~/ray_env \
    --script=train.py -- --num-gpus=2 --num-nodes=1
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
```

### Inference

Give the classifier an image — it returns the top-3 predicted CIFAR-10 categories with confidence scores:

```bash
# Test on CIFAR-10 validation samples
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=image_classifier.py -- \
    --model ../models/cifar10_resnet18_best.pth --image photo.jpg
```

---

## Example 2: GPT-2

### About the dataset

[OpenWebText](https://huggingface.co/datasets/openwebtext) is a reproduction of the WebText corpus used to train the original GPT-2. It contains ~8 million web pages scraped from URLs shared on Reddit with at least 3 karma. The text is tokenized with GPT-2's BPE tokenizer (50,257 token vocabulary) into ~4.5 billion tokens. A 0.05% held-out validation split is used for perplexity evaluation.

### Model sizes

Five model sizes are available via `--model-size`. Larger models generate proportionally more network traffic per step — use them to stress-test the IB fabric.

| Size | Layers | Heads | Dim | Params | Allreduce/step | Recommended mode |
|------|--------|-------|-----|--------|----------------|------------------|
| `small` | 12 | 12 | 768 | 117M | ~0.5 GB | DDP |
| `medium` | 24 | 16 | 1024 | 345M | ~1.4 GB | DDP |
| `large` | 36 | 20 | 1280 | 774M | ~3.1 GB | DDP |
| `xl` | 48 | 25 | 1600 | 1.5B | ~6.0 GB | DDP or FSDP |
| `2b` | 56 | 32 | 2048 | 2.7B | ~10.8 GB | FSDP |

### What the training does

Trains GPT-2 to predict the next token in a sequence. Uses DDP or FSDP for distributed training with gradient accumulation (effective batch = batch_size × grad_accum × num_gpus sequences). Mixed precision (bfloat16 on H200/H100) with cosine LR schedule and warmup. Gradient data is synchronized across all GPUs via allreduce (DDP) or reduce-scatter/allgather (FSDP) — this creates sustained IB traffic that scales with model size.

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
# Validation run — GPT-2 small, 2000 iters
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=2000 --batch-size=8 --grad-accum=8 --save-models

# Full run — 50K iters
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --mode=ddp \
    --max-iters=50000 --batch-size=8 --grad-accum=8 --save-models
```

#### IB stress tests (larger models)

Larger model sizes generate significantly more cross-node traffic. Use FSDP for `2b` — it shards parameters across GPUs, adding allgather + reduce-scatter traffic on top of gradient sync.

```bash
# GPT-2 XL (1.5B) — DDP, 64 GPUs
./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=gpt2_xl \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --model-size=xl --mode=ddp \
    --max-iters=500 --batch-size=4 --grad-accum=4

# GPT-2 2.7B — FSDP, 64 GPUs (heaviest IB load)
./submit_ray_job.sh 8 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=gpt2_2b \
    --script=gpt2_distributed_training.py -- \
    --num-gpus=64 --num-nodes=8 --model-size=2b --mode=fsdp \
    --max-iters=500 --batch-size=2 --grad-accum=8
```

#### Batch size reference by model size (H200, 141 GB)

| Size | DDP per-GPU batch | FSDP per-GPU batch |
|------|-------------------|--------------------|
| small (117M) | 8 | 8 |
| medium (345M) | 8 | 8 |
| large (774M) | 4–8 | 8 |
| xl (1.5B) | 2–4 | 4–8 |
| 2b (2.7B) | — | 2–4 |

### Evaluate & Generate

**Perplexity evaluation** — measures how well the model predicts held-out text (lower = better, expect ~18-20):

```bash
./run_inference.sh --venv=~/ray_env --script=gpt2_eval.py -- \
    --model ../models/gpt2_small_ddp_best.pth --num-batches 200

# For larger models, specify --model-size to match the checkpoint
./run_inference.sh --venv=~/ray_env --script=gpt2_eval.py -- \
    --model ../models/gpt2_xl_ddp_best.pth --model-size=xl --num-batches 200
```

**Text generation** — GPT-2 is a text completion model, not a chatbot. Give it the start of a sentence and it continues writing in the style of web articles:

```bash
./run_inference.sh --venv=~/ray_env --script=gpt2_generate.py -- \
    --model ../models/gpt2_small_ddp_best.pth --prompt "The brain"
```

**Interactive mode** — start a GPU shell and run the generate script directly:

```bash
bsub -n8 -q gpu_l4 -gpu "num=1" -W 4:00 -Is /bin/bash
# once on the GPU node:
cd ~/Ray_IB/ray-training-examples
source ~/ray_env/bin/activate
python gpt2_generate.py --model ../models/gpt2_small_ddp_best.pth --interactive
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

Give the classifier an image — it returns the top-5 predicted ImageNet categories (out of 1,000) with confidence scores:

```bash
# Test on ImageNet validation samples
./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=imagenet_classifier.py -- \
    --model ../models/resnet50_imagenet_best.pth --image photo.jpg
```

### Batch size reference

LR is automatically scaled: `lr = base_lr * effective_batch / 256`.

| GPU | Per-GPU batch |
|-----|--------------|
| H200 (141 GB) | 128–256 |
| H100 (80 GB) | 128 |

---

## Example 4: ViT on ImageNet-1K

### About the model

The Vision Transformer (ViT) applies the transformer architecture — the same architecture behind GPT and LLaMA — to image classification. Instead of convolutional layers (ResNet), ViT splits images into 16x16 pixel patches, treats each patch as a token, and processes them through transformer layers with self-attention. This gives every patch global context from the first layer.

Three model sizes are available via `--model-size`:

| Size | Layers | Heads | Dim | Params | Allreduce/step |
|------|--------|-------|-----|--------|----------------|
| `base` | 12 | 12 | 768 | 86M | ~0.3 GB |
| `large` | 24 | 16 | 1024 | 307M | ~1.2 GB |
| `huge` | 32 | 16 | 1280 | 632M | ~2.5 GB |

### What the training does

Trains ViT on ImageNet-1K using DDP with AdamW optimizer (standard for transformers, unlike SGD for ResNet). Learning rate is scaled linearly by effective batch size / 1024 with linear warmup and cosine decay. Gradient clipping (1.0) is applied for training stability. Same ImageNet data pipeline as the ResNet-50 example.

### What it produces

- Training logs with per-epoch top-1 accuracy, loss, and images/sec throughput
- Model checkpoints (`.pth` files) saved to `../models/`
- Trained models can be used with `vit_imagenet_classifier.py` for inference

### Data

Same ImageNet-1K dataset as the ResNet-50 example — pre-installed at `/nrs/ml_datasets/imagenet`.

### Train

```bash
# ViT-Large, 2 nodes, 16 GPUs — validation run
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=vit_large \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --model-size=large --epochs=3 --batch-size=32 --save-models

# ViT-Large, full 90 epochs
./submit_ray_job.sh 2 --queue=gpu_h200_parallel --venv=~/ray_env \
    --job-name=vit_large \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=16 --num-nodes=2 --model-size=large --epochs=90 --batch-size=32 --save-models

# ViT-Huge, 7 nodes, 56 GPUs — heaviest vision workload
./submit_ray_job.sh 7 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=vit_huge \
    --script=vit_imagenet_distributed_training.py -- \
    --num-gpus=56 --num-nodes=7 --model-size=huge --epochs=90 --batch-size=16 --save-models
```

### Inference

```bash
# Test on ImageNet validation samples
./run_inference.sh --venv=~/ray_env --script=vit_imagenet_classifier.py -- \
    --model ../models/vit_large_imagenet_best.pth --test

# Classify your own image
./run_inference.sh --venv=~/ray_env --script=vit_imagenet_classifier.py -- \
    --model ../models/vit_large_imagenet_best.pth --image photo.jpg
```

### Batch size reference (ViT-Large)

| GPU | Per-GPU batch |
|-----|--------------|
| H200 (141 GB) | 64–128 |
| H100 (80 GB) | 32–64 |
| A100 (40/80 GB) | 16–32 |

---

## Example 5: Cell Segmentation (Swin + LIVECell)

### About the example

This example demonstrates **transfer learning** — the practical workflow most researchers actually use. Instead of training from scratch, we take a Swin Transformer pretrained on ImageNet and fine-tune it for cell segmentation on microscopy images.

The model predicts three classes per pixel: background, cell interior, and cell boundary. Post-processing (watershed) converts the predictions into individual cell instances with colored masks.

### About the dataset

[LIVECell](https://www.nature.com/articles/s41592-021-01249-6) is 5,239 phase contrast microscopy images with 1.6 million manually annotated cell boundaries across 8 cell types (A172, BT474, BV2, Huh7, MCF7, SHSY5Y, SkBr3, SKOV3). Published in Nature Methods.

### What it demonstrates

- **Transfer learning:** Pretrained ImageNet backbone → fine-tune on domain data
- **Differential learning rates:** Backbone gets 10x lower LR than the new segmentation head
- **Segmentation pipeline:** From raw microscopy image to colored cell masks
- **The pattern researchers would use:** swap LIVECell for your own microscopy data

### Data

LIVECell is pre-installed at `/nrs/ml_datasets/livecell` (shared NFS, ~2 GB). Precomputed semantic masks are at `/nrs/ml_datasets/livecell/masks/`.

<details><summary>Preparing your own copy (optional)</summary>

```bash
# Download dataset
./prepare_livecell.sh --data-dir=/your/path

# Precompute semantic masks (one-time, uses all CPU cores, ~5 min)
python prepare_livecell_masks.py --data-dir=/your/path
```

The mask preprocessing converts COCO instance annotations into per-pixel semantic masks (background/cell/boundary) and saves them as `.npy` files. This must be done once before training.

</details>

### Train

```bash
# Fine-tune on 2 H100 nodes (16 GPUs) — 50 epochs
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=livecell \
    --script=livecell_finetune.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=4 --save-models

# Single A100 node (4 GPUs) — good for development
./submit_ray_job.sh 1 --queue=gpu_a100_parallel --venv=~/ray_env \
    --job-name=livecell \
    --script=livecell_finetune.py -- \
    --num-gpus=4 --num-nodes=1 --epochs=50 --batch-size=4 --save-models

# Train from scratch (no pretrained backbone) — for comparison
./submit_ray_job.sh 2 --queue=gpu_h100_parallel --venv=~/ray_env \
    --job-name=livecell_scratch \
    --script=livecell_finetune.py -- \
    --num-gpus=16 --num-nodes=2 --epochs=50 --batch-size=4 --save-models --no-pretrained
```

### Inference

The segmenter takes a microscopy image and produces a colored overlay with individual cell instances:

```bash
# Test on LIVECell test images
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --test

# Segment your own microscopy image
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --image cells.tif

# Process a directory of images
./run_inference.sh --venv=~/ray_env --script=cell_segmenter.py -- \
    --model ../models/swin_livecell_best.pth --image-dir ./my_images/
```

Output segmented images are saved to `../output/segmentation/`.

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
    --num-gpus=16 --num-nodes=2 --model-size=small --mode=ddp \
    --max-iters=5000 --save-models \
    --resume=../models/gpt2_small_ddp_latest.pth
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
| NCCL errors or hangs | Contact SCS |
