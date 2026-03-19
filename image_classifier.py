#!/usr/bin/env python3
"""
CIFAR-10 inference — classify images using a trained ResNet-18 checkpoint.

Usage:
    python image_classifier.py --model ./models/cifar10_resnet18_best.pth --test
    python image_classifier.py --model ./models/cifar10_resnet18_best.pth --image photo.jpg
"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from resnet18_cifar import ResNet18

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

DATA_DIR = "/nrs/scicompsys/Goran/cifar10"


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ResNet18()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    acc = ckpt.get("test_accuracy", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {path}")
    print(f"  Epoch: {epoch}  |  Test accuracy: {acc}%")
    return model


def predict(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    top_p, top_i = torch.topk(probs, 3)
    return [(CLASSES[top_i[j]], top_p[j].item()) for j in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", default=None, help="Image file to classify")
    parser.add_argument("--test", action="store_true",
                        help="Run on CIFAR-10 test samples")
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
            print(f"  {i}. {cls:<12} {conf*100:.1f}%")

    elif args.test:
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=False,
            transform=transforms.ToTensor())
        correct = total = 0
        indices = list(range(0, 100, 10))
        for idx in indices:
            img_t, label = testset[idx]
            img = transforms.ToPILImage()(img_t)
            preds = predict(model, img, device)
            true_cls = CLASSES[label]
            ok = preds[0][0] == true_cls
            correct += ok
            total += 1
            mark = "OK" if ok else "  "
            print(f"[{mark}] {true_cls:<12} -> {preds[0][0]} ({preds[0][1]*100:.1f}%)")
        print(f"\nAccuracy: {correct}/{total}")

    else:
        print("Specify --image PATH or --test")


if __name__ == "__main__":
    main()
