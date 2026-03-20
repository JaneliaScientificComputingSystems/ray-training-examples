#!/usr/bin/env python3
"""
Cell segmentation fine-tuning on LIVECell using a pretrained Swin Transformer.
Demonstrates transfer learning: ImageNet-pretrained backbone + segmentation head.
Uses Ray Data for streaming data loading (official Ray Train pattern).

The model predicts 3 classes per pixel: background, cell interior, cell boundary.
Post-processing (watershed) converts predictions to individual cell instances.
"""
import argparse
import glob
import json
import os
import time
import math
import numpy as np
import ray
import ray.train
import ray.train.torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer


# ---------------------------------------------------------------------------
# Segmentation model: Swin backbone + FPN decoder
# ---------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Simple FPN-style decoder for semantic segmentation."""
    def __init__(self, in_channels_list, num_classes=3):
        super().__init__()
        hidden = 256
        # Lateral connections from backbone feature maps
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, hidden, 1) for c in in_channels_list
        ])
        # Smoothing convolutions
        self.smooths = nn.ModuleList([
            nn.Conv2d(hidden, hidden, 3, padding=1) for _ in in_channels_list
        ])
        # Final prediction
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, features):
        # Build FPN top-down
        laterals = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode="bilinear",
                align_corners=False)
        smoothed = [s(l) for s, l in zip(self.smooths, laterals)]
        # Use finest resolution
        out = smoothed[0]
        return self.head(out)


class SwinSegModel(nn.Module):
    """Swin Transformer backbone + segmentation head."""
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()
        if pretrained:
            weights = torchvision.models.Swin_B_Weights.IMAGENET1K_V1
            backbone = torchvision.models.swin_b(weights=weights)
        else:
            backbone = torchvision.models.swin_b(weights=None)

        # Swin features module extracts multi-scale feature maps
        self.features = backbone.features
        self.norm = backbone.norm
        self.permute = backbone.permute

        # Get channel dimensions for each stage
        # Swin-B: [128, 256, 512, 1024] for stages 0-3
        self.in_channels = [128, 256, 512, 1024]
        self.head = SegmentationHead(self.in_channels, num_classes)

    def forward(self, x):
        # Extract multi-scale features from Swin stages
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [1, 3, 5, 7]:  # After each Swin stage
                # x shape: (B, H, W, C) -> (B, C, H, W)
                features.append(x.permute(0, 3, 1, 2))

        # Decode to segmentation map
        logits = self.head(features)
        return logits


# ---------------------------------------------------------------------------
# Data loading: COCO annotations -> semantic masks
# ---------------------------------------------------------------------------

def coco_to_semantic_mask(annotation_file, image_dir):
    """Load COCO annotations and build an index for fast lookup."""
    with open(annotation_file) as f:
        coco = json.load(f)

    # Build image_id -> annotations mapping
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Build image_id -> file_name mapping
    img_info = {img["id"]: img for img in coco["images"]}

    return coco, img_anns, img_info


def rle_to_mask(rle, h, w):
    """Decode COCO RLE to binary mask."""
    if isinstance(rle, dict) and "counts" in rle:
        counts = rle["counts"]
        if isinstance(counts, str):
            # Compressed RLE
            import pycocotools.mask as mask_util
            mask = mask_util.decode(rle)
            return mask
        else:
            # Uncompressed RLE
            mask = np.zeros(h * w, dtype=np.uint8)
            pos = 0
            for i, count in enumerate(counts):
                if i % 2 == 1:
                    mask[pos:pos + count] = 1
                pos += count
            return mask.reshape((h, w), order='F')
    return np.zeros((h, w), dtype=np.uint8)


def polygon_to_mask(segmentation, h, w):
    """Convert COCO polygon annotation to binary mask."""
    import pycocotools.mask as mask_util
    rles = mask_util.frPyObjects(segmentation, h, w)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle)


def create_semantic_mask(annotations, h, w):
    """Convert instance annotations to semantic mask.
    Classes: 0=background, 1=cell interior, 2=cell boundary.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    boundary_width = 2

    for ann in annotations:
        seg = ann.get("segmentation")
        if seg is None:
            continue

        if isinstance(seg, dict):
            # RLE format
            inst_mask = rle_to_mask(seg, h, w)
        elif isinstance(seg, list):
            # Polygon format
            inst_mask = polygon_to_mask(seg, h, w)
        else:
            continue

        # Erode to get interior, difference is boundary
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(inst_mask, iterations=boundary_width).astype(np.uint8)
        boundary = inst_mask - interior

        # Paint: boundary overwrites interior (handles overlapping cells)
        mask[interior > 0] = 1
        mask[boundary > 0] = 2

    return mask


# ---------------------------------------------------------------------------
# Ray Data transforms
# ---------------------------------------------------------------------------

IMG_SIZE = 512  # Train at 512x512 crops


def transform_train_batch(batch):
    """Transform batch: load image + mask, random crop, augment."""
    from PIL import Image
    import io

    images = []
    masks = []
    for img_bytes, mask_bytes in zip(batch["image_bytes"], batch["mask_bytes"]):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
            batch["height"][0], batch["width"][0])

        # Random crop
        img_t = TF.to_tensor(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        _, h, w = img_t.shape
        if h >= IMG_SIZE and w >= IMG_SIZE:
            i = torch.randint(0, h - IMG_SIZE + 1, (1,)).item()
            j = torch.randint(0, w - IMG_SIZE + 1, (1,)).item()
            img_t = img_t[:, i:i+IMG_SIZE, j:j+IMG_SIZE]
            mask_t = mask_t[:, i:i+IMG_SIZE, j:j+IMG_SIZE]
        else:
            img_t = F.interpolate(img_t.unsqueeze(0), (IMG_SIZE, IMG_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t.unsqueeze(0), (IMG_SIZE, IMG_SIZE),
                                   mode="nearest").squeeze(0)

        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img_t = TF.hflip(img_t)
            mask_t = TF.hflip(mask_t)

        # Normalize
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        images.append(img_t)
        masks.append(mask_t.squeeze(0).long())

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
    }


def transform_val_batch(batch):
    """Transform batch: load image + mask, center crop."""
    from PIL import Image
    import io

    images = []
    masks = []
    for img_bytes, mask_bytes in zip(batch["image_bytes"], batch["mask_bytes"]):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
            batch["height"][0], batch["width"][0])

        img_t = TF.to_tensor(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        _, h, w = img_t.shape

        # Center crop or resize
        if h >= IMG_SIZE and w >= IMG_SIZE:
            i = (h - IMG_SIZE) // 2
            j = (w - IMG_SIZE) // 2
            img_t = img_t[:, i:i+IMG_SIZE, j:j+IMG_SIZE]
            mask_t = mask_t[:, i:i+IMG_SIZE, j:j+IMG_SIZE]
        else:
            img_t = F.interpolate(img_t.unsqueeze(0), (IMG_SIZE, IMG_SIZE),
                                  mode="bilinear", align_corners=False).squeeze(0)
            mask_t = F.interpolate(mask_t.unsqueeze(0), (IMG_SIZE, IMG_SIZE),
                                   mode="nearest").squeeze(0)

        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        images.append(img_t)
        masks.append(mask_t.squeeze(0).long())

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
    }


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def dice_loss(pred, target, num_classes=3):
    """Dice loss for segmentation."""
    pred_soft = F.softmax(pred, dim=1)
    loss = 0
    for c in range(num_classes):
        p = pred_soft[:, c]
        t = (target == c).float()
        intersection = (p * t).sum()
        loss += 1 - (2 * intersection + 1) / (p.sum() + t.sum() + 1)
    return loss / num_classes


def compute_iou(pred, target, num_classes=3):
    """Mean IoU across classes."""
    pred_cls = pred.argmax(dim=1)
    ious = []
    for c in range(num_classes):
        p = (pred_cls == c)
        t = (target == c)
        intersection = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch, miou, config,
                    is_best=False, world_rank=0):
    if world_rank != 0:
        return
    model_dir = config.get("model_dir", "./models")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "best" if is_best else "latest"
    path = os.path.join(model_dir, f"swin_livecell_{suffix}.pth")
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mean_iou": miou,
    }, path)
    print(f"Checkpoint saved: {path}  (mIoU={miou:.4f})")


def load_checkpoint(path, model, optimizer, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_miou = ckpt.get("mean_iou", 0)
    print(f"Resumed from epoch {start_epoch}, best mIoU {best_miou:.4f}")
    return start_epoch, best_miou


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_func(config):
    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    # 1. Create model — pretrained Swin backbone + random segmentation head
    model = SwinSegModel(pretrained=config.get("pretrained", True), num_classes=3)
    model = ray.train.torch.prepare_model(
        model, ddp_kwargs={"find_unused_parameters": True})

    device = ray.train.torch.get_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    if world_rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        raw = model.module if hasattr(model, "module") else model
        backbone_n = sum(p.numel() for p in raw.features.parameters())
        print(f"Cluster: {world_size} GPUs across {config['num_nodes']} nodes")
        print(f"GPU: {torch.cuda.get_device_name(device)} | AMP: {dtype}")
        print(f"Swin-B + SegHead: {n_params/1e6:.1f}M total params")
        print(f"Pretrained backbone: {config.get('pretrained', True)}")

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    # Lower LR for pretrained backbone, higher for new head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config["lr"] * 0.1},  # backbone: 10x lower
        {"params": head_params, "lr": config["lr"]},             # head: full LR
    ], weight_decay=config.get("weight_decay", 0.01))

    criterion = nn.CrossEntropyLoss()

    best_miou = 0
    start_epoch = 0

    if config.get("resume_checkpoint") and world_rank == 0:
        start_epoch, best_miou = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, device)

    if world_size > 1:
        import torch.distributed as dist
        se = torch.tensor(start_epoch, device=device)
        bm = torch.tensor(best_miou, device=device)
        dist.broadcast(se, src=0)
        dist.broadcast(bm, src=0)
        start_epoch = int(se.item())
        best_miou = bm.item()

    for epoch in range(start_epoch, config["epochs"]):
        model.train()

        # Cosine LR with warmup
        warmup = config.get("warmup_epochs", 5)
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(1, config["epochs"] - warmup)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale if "initial_lr" in pg else pg["lr"]
        # Store initial LR on first epoch
        if epoch == 0:
            for pg in optimizer.param_groups:
                pg["initial_lr"] = pg["lr"]

        epoch_loss = 0
        num_batches = 0
        t0 = time.time()

        train_dataloader = train_shard.iter_torch_batches(
            batch_size=config["batch_size"],
            prefetch_batches=2,
        )

        for batch in train_dataloader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=dtype):
                logits = model(images)
                # Upsample logits to mask size
                logits = F.interpolate(logits, size=masks.shape[1:],
                                       mode="bilinear", align_corners=False)
                loss = criterion(logits, masks) + dice_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            if world_rank == 0 and num_batches % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch} [{num_batches}] "
                      f"loss={loss.item():.4f} "
                      f"{num_batches * config['batch_size'] * world_size / elapsed:.0f} img/s")

        train_loss = epoch_loss / max(num_batches, 1)

        # Save latest
        if config.get("save_models"):
            save_checkpoint(model, optimizer, epoch, best_miou,
                            config, is_best=False, world_rank=world_rank)

        # Validation
        val_miou = 0
        if (epoch + 1) % config.get("eval_interval", 5) == 0 or epoch == config["epochs"] - 1:
            model.eval()
            val_ious = []

            val_dataloader = val_shard.iter_torch_batches(
                batch_size=config["batch_size"],
                prefetch_batches=2)

            with torch.no_grad():
                for batch in val_dataloader:
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", dtype=dtype):
                        logits = model(images)
                        logits = F.interpolate(logits, size=masks.shape[1:],
                                               mode="bilinear", align_corners=False)
                    val_ious.append(compute_iou(logits, masks))

            val_miou = np.mean(val_ious) if val_ious else 0.0

            if world_size > 1:
                import torch.distributed as dist
                vm = torch.tensor(val_miou, device=device)
                dist.all_reduce(vm, op=dist.ReduceOp.AVG)
                val_miou = vm.item()

            if config.get("save_models") and val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(model, optimizer, epoch, val_miou,
                                config, is_best=True, world_rank=world_rank)

        ray.train.report({
            "epoch": epoch, "train_loss": train_loss,
            "val_miou": val_miou, "best_miou": best_miou,
        })

        if world_rank == 0:
            val_str = f"val_mIoU={val_miou:.4f}" if val_miou > 0 else ""
            elapsed = time.time() - t0
            print(f"Epoch {epoch}: loss={train_loss:.4f} {val_str} "
                  f"{elapsed:.0f}s")

    if world_rank == 0:
        print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")


# ---------------------------------------------------------------------------
# Data preparation: precompute semantic masks from COCO annotations
# ---------------------------------------------------------------------------

def prepare_dataset(data_dir, split="train"):
    """Build a Ray Dataset from LIVECell images + precomputed masks.
    Masks must be precomputed with prepare_livecell_masks.py first.
    """
    ann_file = os.path.join(data_dir, f"livecell_coco_{split}.json")
    mask_dir = os.path.join(data_dir, "masks", split)
    image_dir = os.path.join(data_dir, "images", "livecell_train_val_images"
                             if split in ("train", "val")
                             else "livecell_test_images")

    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(
            f"Precomputed masks not found at {mask_dir}\n"
            f"Run: python prepare_livecell_masks.py --data-dir={data_dir}")

    print(f"Loading {split} from {image_dir} + {mask_dir}...")
    with open(ann_file) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}

    records = []
    for img_id, info in img_info.items():
        fname = info["file_name"]
        fpath = os.path.join(image_dir, fname)
        base = os.path.splitext(fname)[0]
        mpath = os.path.join(mask_dir, f"{base}.npy")

        if not os.path.exists(fpath) or not os.path.exists(mpath):
            continue

        h, w = info["height"], info["width"]

        with open(fpath, "rb") as f:
            img_bytes = f.read()

        mask = np.load(mpath)
        mask_bytes = mask.tobytes()

        records.append({
            "image_bytes": img_bytes,
            "mask_bytes": mask_bytes,
            "height": h,
            "width": w,
            "file_name": fname,
        })

    print(f"  {split}: {len(records)} images with masks")
    return ray.data.from_items(records)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus",    type=int, required=True)
    parser.add_argument("--num-nodes",   type=int, required=True)
    parser.add_argument("--batch-size",  type=int, default=4,
                        help="per-GPU micro-batch size")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="learning rate for segmentation head (backbone gets 10x lower)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch (no ImageNet pretrained weights)")
    parser.add_argument("--data-dir",    type=str,
                        default="/nrs/ml_datasets/livecell")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, "livecell_coco_train.json")):
        print(f"ERROR: LIVECell annotations not found at {data_dir}")
        print("Download from: http://livecell-dataset.s3.eu-central-1.amazonaws.com/")
        return

    model_dir = os.path.join(os.path.abspath(os.getcwd()), "..", "models")

    ray.init(address="auto")
    print(f"Ray resources: {ray.available_resources()}")

    nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
    print(f"NCCL backend: {'InfiniBand' if nccl_ib == '0' else 'Ethernet'}")
    print(f"Pretrained backbone: {not args.no_pretrained}")

    # Load datasets
    train_ds = prepare_dataset(data_dir, "train")
    val_ds = prepare_dataset(data_dir, "val")

    train_ds = train_ds.map_batches(transform_train_batch)
    val_ds = val_ds.map_batches(transform_val_batch)

    scaling_config = ScalingConfig(
        num_workers=args.num_gpus,
        use_gpu=True,
        resources_per_worker={"CPU": 7, "GPU": 1},
    )
    run_config = RunConfig(
        name="swin_livecell",
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
            "eval_interval": args.eval_interval,
            "pretrained":    not args.no_pretrained,
            "save_models":       args.save_models,
            "resume_checkpoint": args.resume,
            "model_dir":         model_dir,
        },
        datasets={"train": train_ds, "val": val_ds},
        scaling_config=scaling_config,
        run_config=run_config,
    )

    print(f"Starting Swin-B LIVECell fine-tuning: "
          f"{args.num_gpus} GPUs, {args.epochs} epochs")
    result = trainer.fit()

    if result and result.metrics:
        print(f"Best mIoU: {result.metrics.get('best_miou', 0):.4f}")

    ray.shutdown()


if __name__ == "__main__":
    main()
