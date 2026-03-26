#!/usr/bin/env python3
"""
Vision Transformer (ViT) distributed training on ImageNet-1K using torchrun.
Direct PyTorch DDP — no Ray dependency. Equivalent to vit_imagenet_distributed_training.py.

Supports base (86M), large (307M), and huge (632M) model sizes.
Uses PyTorch DistributedSampler + DataLoader over HuggingFace parquet shards.

Usage (via submit_torchrun_job.sh):
    ./submit_torchrun_job.sh 2 --queue=gpu_h100_parallel \
        --script=vit_imagenet_torchrun.py --venv=~/ray_env \
        -- --model-size=base --epochs=5 --batch-size=64

    ./submit_torchrun_job.sh 1 --queue=gpu_h100 --num-gpus=4 \
        --script=vit_imagenet_torchrun.py --venv=~/ray_env \
        -- --model-size=large --epochs=90
"""
import argparse
import glob
import io
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

VIT_MODELS = {
    "base":  {"fn": torchvision.models.vit_b_16,  "params": "86M"},
    "large": {"fn": torchvision.models.vit_l_16,  "params": "307M"},
    "huge":  {"fn": torchvision.models.vit_h_14,  "params": "632M"},
}


# ---------------------------------------------------------------------------
# Dataset — reads HuggingFace ImageNet parquet shards
# ---------------------------------------------------------------------------

class ImageNetParquetDataset(Dataset):
    """Reads ImageNet from HuggingFace parquet files (image bytes + label)."""

    def __init__(self, parquet_files, transform=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.transform = transform
        # Read all parquet files into a single table
        tables = [pq.read_table(f, columns=["image", "label"]) for f in parquet_files]
        table = pa.concat_tables(tables)
        self.images = table.column("image").to_pylist()  # list of {"bytes": ..., "path": ...}
        self.labels = table.column("label").to_pylist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Transforms (same as Ray version)
# ---------------------------------------------------------------------------

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


# ---------------------------------------------------------------------------
# LR schedule — linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k * 100.0 / batch_size)
        return res


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, top1, top5, args, is_best=False):
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    suffix = "best" if is_best else "latest"
    path = os.path.join(model_dir, f"vit_{args.model_size}_imagenet_torchrun_{suffix}.pth")
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "model_size": args.model_size,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
    }, path)
    print(f"Checkpoint saved: {path}  (top1={top1:.2f}% top5={top5:.2f}%)")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_top1 = ckpt.get("top1_accuracy", 0)
    print(f"Resumed from epoch {start_epoch}, best top1 {best_top1:.2f}%")
    return start_epoch, best_top1


# ---------------------------------------------------------------------------
# IB verification (same as Ray version)
# ---------------------------------------------------------------------------

def verify_ib_in_use():
    import subprocess
    ib_devs = ["mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3",
               "mlx5_4", "mlx5_6", "mlx5_7", "mlx5_8"]
    eth_dev = "mlx5_5"
    print("--- IB device check ---")
    for dev in ib_devs:
        try:
            r = subprocess.run(["ibv_devinfo", "-d", dev],
                               capture_output=True, text=True, timeout=5)
            link = next((l.strip() for l in r.stdout.splitlines()
                         if "link_layer" in l), "unknown")
            state = next((l.strip() for l in r.stdout.splitlines()
                          if "state:" in l), "unknown")
            ok = "InfiniBand" in link
            print(f"  {'OK  ' if ok else 'WARN'} {dev}: {link} | {state}")
        except Exception as e:
            print(f"  ERR  {dev}: {e}")
    nccl_hca = os.environ.get("NCCL_IB_HCA", "")
    if eth_dev in nccl_hca:
        print(f"  ERROR: {eth_dev} (Ethernet) is in NCCL_IB_HCA — remove it!")
    else:
        print(f"  OK   {eth_dev} (Ethernet) correctly excluded from NCCL_IB_HCA")
    print("-----------------------")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=list(VIT_MODELS.keys()), default="base",
                        help="ViT size: base=86M, large=307M, huge=632M")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="per-GPU micro-batch size")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="base LR (AdamW) — scaled linearly by effective batch / 1024")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="linear LR warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="evaluate every N epochs")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data-dir", type=str,
                        default="/nrs/ml_datasets/imagenet")
    parser.add_argument("--model-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # torchrun sets these
    local_rank = int(os.environ["LOCAL_RANK"])
    world_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Init process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.model_dir is None:
        args.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "models")

    if world_rank == 0:
        print("================================================================")
        print("ViT ImageNet Distributed Training (torchrun)")
        print("================================================================")
        nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
        backend_name = "InfiniBand (GPUDirect RDMA)" if nccl_ib == "0" else "Ethernet"
        print(f"Network: {backend_name}")
        if nccl_ib == "0":
            verify_ib_in_use()

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    parquet_dir = os.path.join(args.data_dir, "data")
    if not os.path.isdir(parquet_dir):
        parquet_dir = args.data_dir

    train_files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(parquet_dir, "validation-*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train parquet files in {parquet_dir}")

    if world_rank == 0:
        print(f"Loading ImageNet: {len(train_files)} train shards, "
              f"{len(val_files)} val shards")
        t_load = time.time()

    train_dataset = ImageNetParquetDataset(train_files, transform=train_transform)
    val_dataset = ImageNetParquetDataset(val_files, transform=val_transform)

    if world_rank == 0:
        print(f"Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val "
              f"images in {time.time() - t_load:.1f}s")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                       rank=world_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                     rank=world_rank, shuffle=False)

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(8, max(4, cpu_count // 8))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True)

    # -----------------------------------------------------------------------
    # Model + DDP
    # -----------------------------------------------------------------------
    model_info = VIT_MODELS[args.model_size]
    model = model_info["fn"](weights=None, num_classes=1000).to(device)
    model = DDP(model, device_ids=[local_rank])

    n_params = sum(p.numel() for p in model.parameters())
    effective_batch = args.batch_size * world_size
    scaled_lr = args.lr * effective_batch / 1024

    # AMP
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    if world_rank == 0:
        print(f"Model: ViT-{args.model_size} ({n_params/1e6:.1f}M params)")
        print(f"GPUs: {world_size} | AMP: {dtype}")
        print(f"Per-GPU batch: {args.batch_size} | Effective batch: {effective_batch} "
              f"| Scaled LR: {scaled_lr:.6f}")
        print(f"Epochs: {args.epochs} | Warmup: {args.warmup_epochs} | "
              f"Eval every: {args.eval_interval}")
        print("================================================================")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=scaled_lr,
        betas=(0.9, 0.999), weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0.0
    start_epoch = 0

    if args.resume:
        if world_rank == 0:
            start_epoch, best_top1 = load_checkpoint(
                args.resume, model, optimizer, device)
        # Broadcast resume state
        se = torch.tensor(start_epoch, device=device)
        bt = torch.tensor(best_top1, device=device)
        dist.broadcast(se, src=0)
        dist.broadcast(bt, src=0)
        start_epoch = int(se.item())
        best_top1 = bt.item()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        lr = get_lr(epoch, args.warmup_epochs, args.epochs, scaled_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        images_processed = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                output = model(data)
                loss = criterion(output, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            total += targets.size(0)
            correct += output.argmax(1).eq(targets).sum().item()
            images_processed += data.size(0)

            if world_rank == 0 and (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                img_sec = images_processed * world_size / elapsed
                print(f"  Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} lr={lr:.6f} "
                      f"{img_sec:.0f} img/s")

        num_batches = batch_idx + 1
        train_loss = epoch_loss / max(num_batches, 1)
        train_top1 = 100.0 * correct / max(total, 1)
        elapsed = time.time() - t0
        img_sec = images_processed * world_size / max(elapsed, 1)

        if args.save_models and world_rank == 0:
            save_checkpoint(model, optimizer, epoch, train_top1, 0, args, is_best=False)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        val_top1 = 0.0
        val_top5 = 0.0
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_correct1 = val_correct5 = val_total = 0

            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=dtype):
                        output = model(data)
                    top1, top5 = accuracy(output, targets, topk=(1, 5))
                    val_correct1 += top1.item() * targets.size(0) / 100
                    val_correct5 += top5.item() * targets.size(0) / 100
                    val_total += targets.size(0)

            if val_total > 0:
                val_top1 = 100.0 * val_correct1 / val_total
                val_top5 = 100.0 * val_correct5 / val_total

            # All-reduce val metrics for accurate reporting
            t1 = torch.tensor(val_top1, device=device)
            t5 = torch.tensor(val_top5, device=device)
            dist.all_reduce(t1, op=dist.ReduceOp.AVG)
            dist.all_reduce(t5, op=dist.ReduceOp.AVG)
            val_top1 = t1.item()
            val_top5 = t5.item()

            if args.save_models and val_top1 > best_top1 and world_rank == 0:
                best_top1 = val_top1
                save_checkpoint(model, optimizer, epoch, val_top1, val_top5,
                                args, is_best=True)

        if world_rank == 0:
            val_str = (f"val_top1={val_top1:.2f}% val_top5={val_top5:.2f}%"
                       if val_top1 > 0 else "")
            print(f"Epoch {epoch}: train_top1={train_top1:.2f}% "
                  f"loss={train_loss:.4f} {val_str} "
                  f"lr={lr:.6f} {img_sec:.0f} img/s")

    if world_rank == 0:
        print(f"\nTraining complete. Best top-1 accuracy: {best_top1:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
