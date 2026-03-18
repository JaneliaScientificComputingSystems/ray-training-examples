#!/usr/bin/env python3
"""CIFAR-10 inference — works with checkpoints from any queue."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os

CIFAR10_CLASSES = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1   = nn.Conv2d(3, 64, kernel_size=3,
                                         stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc      = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)

class ImageClassifier:
    def __init__(self, model_path, device='auto'):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') \
            if device == 'auto' else torch.device(device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = ResNet18()
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
        ])
        acc = ckpt.get('test_accuracy', 'unknown')
        print(f"Model loaded | test accuracy: {acc}% | device: {self.device}")

    def predict(self, image_path, top_k=3):
        img = Image.open(image_path).convert('RGB') \
              if isinstance(image_path, str) else image_path.convert('RGB')
        t   = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(self.model(t)[0], dim=0)
        top_p, top_i = torch.topk(probs, top_k)
        return [(CIFAR10_CLASSES[top_i[i]], top_p[i].item()) for i in range(top_k)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  required=True)
    parser.add_argument('--image',  default=None)
    parser.add_argument('--test',   action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    clf = ImageClassifier(args.model, args.device)

    if args.image:
        preds = clf.predict(args.image)
        print(f"\nImage: {args.image}")
        for i, (cls, conf) in enumerate(preds, 1):
            print(f"  {i}. {cls:<12} {conf*100:.1f}%")

    elif args.test:
        import torchvision.datasets as datasets
        data_path = os.path.expanduser("~/datasets/cifar10")
        testset   = datasets.CIFAR10(root=data_path, train=False,
                                      download=False,
                                      transform=transforms.ToTensor())
        correct = 0
        indices = [0, 10, 50, 100, 500]
        for idx in indices:
            img, label = testset[idx]
            true_cls   = CIFAR10_CLASSES[label]
            preds      = clf.predict(transforms.ToPILImage()(img))
            marker     = "OK" if preds[0][0] == true_cls else "  "
            print(f"[{marker}] {true_cls:<12} -> {preds[0][0]} ({preds[0][1]*100:.1f}%)")
            if preds[0][0] == true_cls:
                correct += 1
        print(f"\nAccuracy on sample: {correct}/{len(indices)}")
    else:
        print("Specify --image PATH or --test")

if __name__ == "__main__":
    main()
