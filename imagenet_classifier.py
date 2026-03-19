#!/usr/bin/env python3
"""
ImageNet inference — classify images using a trained ResNet-50 checkpoint.

Usage:
    python imagenet_classifier.py --model ./models/resnet50_imagenet_best.pth --image photo.jpg
    python imagenet_classifier.py --model ./models/resnet50_imagenet_best.pth --test
"""
import argparse
import io
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

DATA_DIR = "/nrs/ml_datasets/imagenet"

# ImageNet class index — loaded lazily from checkpoint dir or torchvision
_CLASS_NAMES = None


def get_class_names():
    """Return list of 1000 ImageNet class names."""
    global _CLASS_NAMES
    if _CLASS_NAMES is not None:
        return _CLASS_NAMES
    # Use torchvision's built-in ImageNet class metadata
    from torchvision.models import ResNet50_Weights
    _CLASS_NAMES = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    return _CLASS_NAMES


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = torchvision.models.resnet50(weights=None, num_classes=1000)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    top1 = ckpt.get("top1_accuracy", "?")
    top5 = ckpt.get("top5_accuracy", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {path}")
    print(f"  Epoch: {epoch}  |  Top-1: {top1}%  |  Top-5: {top5}%")
    return model


def predict(model, image, device, top_k=5):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    top_p, top_i = torch.topk(probs, top_k)
    names = get_class_names()
    return [(names[top_i[j]], top_p[j].item()) for j in range(top_k)]


def run_test(model, device, data_dir, num_samples=20):
    """Evaluate on random validation images from parquet shards."""
    import pyarrow.parquet as pq
    import glob
    import random

    val_files = sorted(glob.glob(os.path.join(data_dir, "data",
                                              "validation-*.parquet")))
    if not val_files:
        val_files = sorted(glob.glob(os.path.join(data_dir,
                                                  "validation-*.parquet")))
    if not val_files:
        print(f"No validation parquet files found in {data_dir}")
        return

    # Read a subset of rows
    table = pq.read_table(random.choice(val_files))
    indices = random.sample(range(len(table)), min(num_samples, len(table)))

    correct = total = 0
    names = get_class_names()
    for idx in indices:
        row = table.slice(idx, 1)
        label = row.column("label")[0].as_py()
        img_data = row.column("image")[0].as_py()
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        preds = predict(model, img, device)
        true_name = names[label]
        ok = preds[0][0] == true_name
        correct += ok
        total += 1
        mark = "OK" if ok else "  "
        print(f"[{mark}] {true_name:<30} -> {preds[0][0]} ({preds[0][1]*100:.1f}%)")

    print(f"\nAccuracy: {correct}/{total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", default=None, help="Image file to classify")
    parser.add_argument("--test", action="store_true",
                        help="Run on ImageNet validation samples")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of test samples (default: 20)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    model = load_model(args.model, device)

    if args.image:
        img = Image.open(args.image).convert("RGB")
        preds = predict(model, img, device)
        print(f"\n{args.image}:")
        for i, (cls, conf) in enumerate(preds, 1):
            print(f"  {i}. {cls:<30} {conf*100:.1f}%")

    elif args.test:
        run_test(model, device, args.data_dir, args.num_samples)

    else:
        print("Specify --image PATH or --test")


if __name__ == "__main__":
    main()
