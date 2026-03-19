#!/usr/bin/env python3
"""
ResNet-50 distributed training on ImageNet-1K (ILSVRC2012).
Industry-standard distributed training benchmark.
Uses Ray Data for streaming data loading (official Ray Train pattern).
Works on all Janelia GPU queues — IB and Ethernet NCCL.
"""
import argparse
import glob
import os
import time
import math
import numpy as np
import ray
import ray.train
import ray.train.torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus",    type=int, required=True)
    parser.add_argument("--num-nodes",   type=int, required=True)
    parser.add_argument("--batch-size",  type=int, default=64,
                        help="per-GPU micro-batch size")
    parser.add_argument("--epochs",      type=int, default=90)
    parser.add_argument("--lr",          type=float, default=0.1,
                        help="base LR — scaled linearly by effective batch size / 256")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="linear LR warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum",    type=float, default=0.9)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="evaluate every N epochs")
    parser.add_argument("--data-dir",    type=str,
                        default="/nrs/ml_datasets/imagenet")
    return parser.parse_args()


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
# Image transforms
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


def transform_train_batch(batch):
    """Transform a Ray Data batch for training."""
    from PIL import Image
    import io
    images = []
    for img_dict in batch["image"]:
        img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        images.append(train_transform(img))
    return {
        "image": torch.stack(images),
        "label": torch.tensor(batch["label"], dtype=torch.long),
    }


def transform_val_batch(batch):
    """Transform a Ray Data batch for validation."""
    from PIL import Image
    import io
    images = []
    for img_dict in batch["image"]:
        img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
        images.append(val_transform(img))
    return {
        "image": torch.stack(images),
        "label": torch.tensor(batch["label"], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, top1, top5, config,
                    is_best=False, world_rank=0):
    if world_rank != 0:
        return
    model_dir = config.get("model_dir", "./models")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "best" if is_best else "latest"
    path = os.path.join(model_dir, f"resnet50_imagenet_{suffix}.pth")
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
    }, path)
    print(f"Checkpoint saved: {path}  (top1={top1:.2f}% top5={top5:.2f}%)")


def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_top1   = ckpt.get("top1_accuracy", 0)
    print(f"Resumed from epoch {start_epoch}, best top1 {best_top1:.2f}%")
    return start_epoch, best_top1


# ---------------------------------------------------------------------------
# Training function (runs on each Ray worker)
# ---------------------------------------------------------------------------

def train_func(config):
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # 1. Create model on CPU, prepare_model handles device + DDP + NCCL
    model = torchvision.models.resnet50(weights=None, num_classes=1000)
    model = ray.train.torch.prepare_model(model)

    # 2. AMP setup after prepare_model
    device = ray.train.torch.get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    if world_rank == 0:
        print(f"Cluster: {world_size} GPUs across {config['num_nodes']} nodes")
        print(f"GPU: {torch.cuda.get_device_name(device)} | AMP: {dtype}")
        effective_batch = config["batch_size"] * world_size
        scaled_lr = config["lr"] * effective_batch / 256
        print(f"Per-GPU batch: {config['batch_size']} | "
              f"Effective batch: {effective_batch} | Scaled LR: {scaled_lr:.4f}")
        nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
        print(f"Network: {'InfiniBand (GPUDirect RDMA)' if nccl_ib == '0' else 'Ethernet'}")

    # 3. Get Ray Data shards — automatically split across workers
    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    # 4. Optimizer
    effective_batch = config["batch_size"] * world_size
    scaled_lr = config["lr"] * effective_batch / 256
    optimizer = torch.optim.SGD(
        model.parameters(), lr=scaled_lr,
        momentum=config["momentum"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_top1 = 0
    start_epoch = 0

    if config.get("resume_checkpoint") and world_rank == 0:
        start_epoch, best_top1 = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, device)

    if world_size > 1:
        import torch.distributed as dist
        se = torch.tensor(start_epoch, device=device)
        bt = torch.tensor(best_top1,   device=device)
        dist.broadcast(se, src=0)
        dist.broadcast(bt, src=0)
        start_epoch = int(se.item())
        best_top1   = bt.item()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        lr = get_lr(epoch, config["warmup_epochs"], config["epochs"], scaled_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        epoch_loss = 0
        correct = 0
        total = 0
        t0 = time.time()
        images_processed = 0
        num_batches = 0

        # Stream batches from Ray Data
        train_dataloader = train_shard.iter_torch_batches(
            batch_size=config["batch_size"],
            prefetch_batches=2,
        )

        for batch in train_dataloader:
            data = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                output = model(data)
                loss = criterion(output, targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            total += targets.size(0)
            correct += output.argmax(1).eq(targets).sum().item()
            images_processed += data.size(0)
            num_batches += 1

            if world_rank == 0 and num_batches % 100 == 0:
                elapsed = time.time() - t0
                img_sec = images_processed * world_size / elapsed
                print(f"  Epoch {epoch} [{num_batches}] "
                      f"loss={loss.item():.4f} lr={lr:.5f} "
                      f"{img_sec:.0f} img/s")

        train_loss = epoch_loss / max(num_batches, 1)
        train_top1 = 100.0 * correct / max(total, 1)
        elapsed = time.time() - t0
        img_sec = images_processed * world_size / max(elapsed, 1)

        # Save latest checkpoint every epoch
        if config.get("save_models"):
            save_checkpoint(model, optimizer, epoch, train_top1, 0,
                            config, is_best=False, world_rank=world_rank)

        # Validation
        val_top1 = 0
        val_top5 = 0
        if (epoch + 1) % config["eval_interval"] == 0 or epoch == config["epochs"] - 1:
            model.eval()
            val_correct1 = val_correct5 = val_total = 0

            val_dataloader = val_shard.iter_torch_batches(
                batch_size=config["batch_size"],
                prefetch_batches=2)

            with torch.no_grad():
                for batch in val_dataloader:
                    data = batch["image"].to(device, non_blocking=True)
                    targets = batch["label"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=dtype):
                        output = model(data)
                    top1, top5 = accuracy(output, targets, topk=(1, 5))
                    val_correct1 += top1.item() * targets.size(0) / 100
                    val_correct5 += top5.item() * targets.size(0) / 100
                    val_total += targets.size(0)

            if val_total > 0:
                val_top1 = 100.0 * val_correct1 / val_total
                val_top5 = 100.0 * val_correct5 / val_total

            if world_size > 1:
                import torch.distributed as dist
                t1 = torch.tensor(val_top1, device=device)
                t5 = torch.tensor(val_top5, device=device)
                dist.all_reduce(t1, op=dist.ReduceOp.AVG)
                dist.all_reduce(t5, op=dist.ReduceOp.AVG)
                val_top1 = t1.item()
                val_top5 = t5.item()

            if config.get("save_models") and val_top1 > best_top1:
                best_top1 = val_top1
                save_checkpoint(model, optimizer, epoch, val_top1, val_top5,
                                config, is_best=True, world_rank=world_rank)

        ray.train.report({
            "epoch": epoch, "train_loss": train_loss,
            "train_top1": train_top1, "val_top1": val_top1,
            "val_top5": val_top5, "lr": lr, "img_sec": img_sec,
            "best_top1": best_top1,
        })

        if world_rank == 0:
            val_str = (f"val_top1={val_top1:.2f}% val_top5={val_top5:.2f}%"
                       if val_top1 > 0 else "")
            print(f"Epoch {epoch}: train_top1={train_top1:.2f}% "
                  f"loss={train_loss:.4f} {val_str} "
                  f"lr={lr:.5f} {img_sec:.0f} img/s")

    if world_rank == 0:
        print(f"\nTraining complete. Best top-1 accuracy: {best_top1:.2f}%")


def main():
    args = parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"ERROR: ImageNet dataset not found at {data_dir}")
        print("Run prepare_imagenet.sh first.")
        return

    submission_dir = os.path.abspath(os.getcwd())
    model_dir = os.path.join(submission_dir, "..", "models")

    ray.init(address="auto")
    print(f"Ray resources: {ray.available_resources()}")

    nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
    print(f"NCCL backend: {'InfiniBand' if nccl_ib == '0' else 'Ethernet'}")

    # Load ImageNet as Ray Datasets from HuggingFace parquet
    print("Loading ImageNet datasets...")
    parquet_dir = os.path.join(data_dir, "data")
    if not os.path.isdir(parquet_dir):
        parquet_dir = data_dir

    train_files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    val_files = sorted(glob.glob(os.path.join(parquet_dir, "validation-*.parquet")))
    if not train_files:
        print(f"ERROR: No train parquet files in {parquet_dir}")
        return
    print(f"Found {len(train_files)} train shards, {len(val_files)} val shards")

    train_ds = ray.data.read_parquet(train_files)
    val_ds = ray.data.read_parquet(val_files)

    # Apply transforms via map_batches
    train_ds = train_ds.map_batches(transform_train_batch)
    val_ds = val_ds.map_batches(transform_val_batch)

    print(f"Train: {len(train_files)} shards | Val: {len(val_files)} shards")

    scaling_config = ScalingConfig(
        num_workers=args.num_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": 7, "GPU": 1},
    )
    run_config = RunConfig(
        name="resnet50_imagenet",
        storage_path="/scratch/{}/ray_results".format(
            os.getenv("USER", "unknown")),
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr":            args.lr,
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "num_nodes":     args.num_nodes,
            "warmup_epochs": args.warmup_epochs,
            "weight_decay":  args.weight_decay,
            "momentum":      args.momentum,
            "eval_interval": args.eval_interval,
            "save_models":       args.save_models,
            "resume_checkpoint": args.resume,
            "model_dir":         model_dir,
        },
        datasets={"train": train_ds, "val": val_ds},
        scaling_config=scaling_config,
        run_config=run_config,
    )

    effective_batch = args.batch_size * args.num_gpus
    print(f"Starting ResNet-50 ImageNet: "
          f"{args.num_gpus} GPUs, effective batch {effective_batch}")
    result = trainer.fit()

    if result and result.metrics:
        print(f"Final top-1: {result.metrics.get('val_top1', 0):.2f}%")
        print(f"Final top-5: {result.metrics.get('val_top5', 0):.2f}%")
        print(f"Best top-1:  {result.metrics.get('best_top1', 0):.2f}%")
        print(f"Throughput:  {result.metrics.get('img_sec', 0):.0f} img/s")

    ray.shutdown()


if __name__ == "__main__":
    main()
