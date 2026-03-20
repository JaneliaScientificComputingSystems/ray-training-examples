#!/usr/bin/env python3
"""
Cell segmentation inference using a trained Swin + LIVECell checkpoint.
Produces segmented images with colored cell masks.

Usage:
    python cell_segmenter.py --model ../models/swin_livecell_best.pth --image cells.tif
    python cell_segmenter.py --model ../models/swin_livecell_best.pth --test
    python cell_segmenter.py --model ../models/swin_livecell_best.pth --image-dir ./my_images/
"""
import argparse
import glob
import io
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Import model definition from training script
from livecell_finetune import SwinSegModel


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = SwinSegModel(pretrained=False, num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    miou = ckpt.get("mean_iou", "?")
    print(f"Loaded: {path}")
    print(f"  Epoch: {epoch}  |  mIoU: {miou}")
    return model


def segment_image(model, image, device, tile_size=512, overlap=64):
    """Segment an image, handling arbitrary sizes via tiling."""
    img_t = TF.to_tensor(image)
    if img_t.shape[0] == 1:
        img_t = img_t.repeat(3, 1, 1)  # Grayscale -> RGB
    img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    _, h, w = img_t.shape
    pred = torch.zeros(3, h, w)
    count = torch.zeros(1, h, w)

    # Tile the image for large inputs
    step = tile_size - overlap
    for y in range(0, max(h, tile_size), step):
        for x in range(0, max(w, tile_size), step):
            y1 = min(y, max(0, h - tile_size))
            x1 = min(x, max(0, w - tile_size))
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            tile = img_t[:, y1:y2, x1:x2].unsqueeze(0).to(device)
            # Pad if smaller than tile_size
            ph = tile_size - tile.shape[2]
            pw = tile_size - tile.shape[3]
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph))

            with torch.no_grad():
                logits = model(tile)
                logits = F.interpolate(logits, size=(tile_size, tile_size),
                                       mode="bilinear", align_corners=False)

            # Crop padding and accumulate
            logits = logits[0, :, :y2-y1, :x2-x1].cpu()
            pred[:, y1:y2, x1:x2] += logits
            count[:, y1:y2, x1:x2] += 1

            if y2 >= h:
                break
        if y + step >= h:
            break

    pred = pred / count.clamp(min=1)
    return pred.argmax(dim=0).numpy()  # (H, W) with class indices


def instances_from_semantic(semantic_mask):
    """Convert semantic segmentation (bg/cell/boundary) to instance labels
    using watershed."""
    from scipy import ndimage
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    cell_mask = (semantic_mask == 1).astype(np.uint8)
    if cell_mask.sum() == 0:
        return np.zeros_like(semantic_mask, dtype=np.int32)

    # Distance transform for watershed seeds
    distance = ndimage.distance_transform_edt(cell_mask)
    coords = peak_local_max(distance, min_distance=10, labels=cell_mask)
    markers = np.zeros_like(cell_mask, dtype=np.int32)
    for i, (y, x) in enumerate(coords, 1):
        markers[y, x] = i

    # Watershed
    labels = watershed(-distance, markers, mask=cell_mask)
    return labels


def colorize_instances(instance_labels):
    """Create a colored RGB image from instance labels."""
    np.random.seed(42)
    max_label = instance_labels.max()
    if max_label == 0:
        return np.zeros((*instance_labels.shape, 3), dtype=np.uint8)

    colors = np.random.randint(50, 255, size=(max_label + 1, 3), dtype=np.uint8)
    colors[0] = 0  # Background is black

    colored = colors[instance_labels]
    return colored


def overlay_on_image(image, instance_labels, alpha=0.4):
    """Overlay colored instance masks on the original image."""
    img_arr = np.array(image)
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[2] == 1:
        img_arr = np.concatenate([img_arr] * 3, axis=-1)

    colored = colorize_instances(instance_labels)
    mask = instance_labels > 0
    result = img_arr.copy()
    result[mask] = (img_arr[mask] * (1 - alpha) + colored[mask] * alpha).astype(np.uint8)

    # Draw boundaries in white
    from scipy.ndimage import find_objects, binary_dilation
    boundary = np.zeros_like(mask)
    for label_id in range(1, instance_labels.max() + 1):
        cell = (instance_labels == label_id)
        dilated = binary_dilation(cell, iterations=1)
        boundary |= (dilated & ~cell)
    result[boundary] = 255

    return result


def process_image(model, image_path, device, output_dir=None):
    """Segment a single image and save/display results."""
    image = Image.open(image_path).convert("RGB")
    print(f"\n{image_path}: {image.size[0]}x{image.size[1]}")

    # Segment
    semantic = segment_image(model, image, device)
    instances = instances_from_semantic(semantic)
    n_cells = instances.max()
    print(f"  Detected {n_cells} cells")

    # Class distribution
    bg = (semantic == 0).sum()
    cell = (semantic == 1).sum()
    boundary = (semantic == 2).sum()
    total = semantic.size
    print(f"  Background: {bg/total*100:.1f}%  Cell: {cell/total*100:.1f}%  "
          f"Boundary: {boundary/total*100:.1f}%")

    # Save output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        overlay = overlay_on_image(image, instances)
        Image.fromarray(overlay).save(os.path.join(output_dir, f"{base}_segmented.png"))
        print(f"  Saved: {output_dir}/{base}_segmented.png")


def run_test(model, device, data_dir, num_samples=5, output_dir=None):
    """Run on random LIVECell test images."""
    import json
    import random

    ann_file = os.path.join(data_dir, "livecell_coco_test.json")
    image_dir = os.path.join(data_dir, "images", "livecell_test_images")

    if not os.path.exists(ann_file):
        # Try train/val images if test not available
        ann_file = os.path.join(data_dir, "livecell_coco_val.json")
        image_dir = os.path.join(data_dir, "images", "livecell_train_val_images")

    if not os.path.exists(ann_file):
        print(f"No annotation files found in {data_dir}")
        return

    with open(ann_file) as f:
        coco = json.load(f)

    images = coco["images"]
    samples = random.sample(images, min(num_samples, len(images)))

    for img_info in samples:
        fpath = os.path.join(image_dir, img_info["file_name"])
        if os.path.exists(fpath):
            process_image(model, fpath, device, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", default=None, help="Single image to segment")
    parser.add_argument("--image-dir", default=None,
                        help="Directory of images to segment")
    parser.add_argument("--test", action="store_true",
                        help="Run on LIVECell test images")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="../output/segmentation",
                        help="Directory to save segmented images")
    parser.add_argument("--data-dir", default="/nrs/ml_datasets/livecell")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    model = load_model(args.model, device)

    if args.image:
        process_image(model, args.image, device, args.output_dir)

    elif args.image_dir:
        for ext in ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"):
            for fpath in sorted(glob.glob(os.path.join(args.image_dir, ext))):
                process_image(model, fpath, device, args.output_dir)

    elif args.test:
        run_test(model, device, args.data_dir, args.num_samples, args.output_dir)

    else:
        print("Specify --image PATH, --image-dir DIR, or --test")


if __name__ == "__main__":
    main()
