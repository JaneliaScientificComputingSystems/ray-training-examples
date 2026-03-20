#!/usr/bin/env python3
"""
Precompute semantic masks for LIVECell dataset.
Converts COCO instance annotations to per-pixel semantic masks
(0=background, 1=cell interior, 2=cell boundary) and saves as .npy files.

Run once before training:
    python prepare_livecell_masks.py
    python prepare_livecell_masks.py --data-dir=/path/to/livecell

Creates masks/ directory alongside images with one .npy per image.
"""
import argparse
import json
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


def polygon_to_mask(segmentation, h, w):
    import pycocotools.mask as mask_util
    rles = mask_util.frPyObjects(segmentation, h, w)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle)


def rle_to_mask(rle, h, w):
    import pycocotools.mask as mask_util
    if isinstance(rle, dict) and "counts" in rle:
        return mask_util.decode(rle)
    return np.zeros((h, w), dtype=np.uint8)


def create_semantic_mask(annotations, h, w, boundary_width=2):
    from scipy.ndimage import binary_erosion
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        if isinstance(seg, dict):
            inst_mask = rle_to_mask(seg, h, w)
        elif isinstance(seg, list):
            inst_mask = polygon_to_mask(seg, h, w)
        else:
            continue
        interior = binary_erosion(inst_mask, iterations=boundary_width).astype(np.uint8)
        boundary = inst_mask - interior
        mask[interior > 0] = 1
        mask[boundary > 0] = 2
    return mask


def process_image(args):
    img_id, info, anns, mask_dir = args
    fname = info["file_name"]
    h, w = info["height"], info["width"]
    base = os.path.splitext(fname)[0]
    out_path = os.path.join(mask_dir, f"{base}.npy")

    if os.path.exists(out_path):
        return fname, "skip"

    mask = create_semantic_mask(anns, h, w)
    np.save(out_path, mask)
    return fname, "done"


def process_split(data_dir, split):
    ann_file = os.path.join(data_dir, f"livecell_coco_{split}.json")
    if not os.path.exists(ann_file):
        print(f"  {split}: annotation file not found, skipping")
        return

    mask_dir = os.path.join(data_dir, "masks", split)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"  Loading {split} annotations...")
    with open(ann_file) as f:
        coco = json.load(f)

    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    img_info = {img["id"]: img for img in coco["images"]}

    tasks = []
    for img_id, info in img_info.items():
        anns = img_anns.get(img_id, [])
        tasks.append((img_id, info, anns, mask_dir))

    done = 0
    skipped = 0
    total = len(tasks)

    print(f"  Processing {total} images ({os.cpu_count()} workers)...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = {pool.submit(process_image, t): t for t in tasks}
        for future in as_completed(futures):
            fname, status = future.result()
            if status == "skip":
                skipped += 1
            else:
                done += 1
            if (done + skipped) % 500 == 0:
                print(f"    {done + skipped}/{total} "
                      f"({done} new, {skipped} cached)")

    print(f"  {split}: {done} masks created, {skipped} already existed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/nrs/ml_datasets/livecell")
    args = parser.parse_args()

    print("================================================================")
    print("LIVECell Mask Preprocessing")
    print(f"Data dir: {args.data_dir}")
    print("================================================================")

    for split in ["train", "val", "test"]:
        process_split(args.data_dir, split)

    total = sum(
        len(os.listdir(os.path.join(args.data_dir, "masks", s)))
        for s in ["train", "val", "test"]
        if os.path.isdir(os.path.join(args.data_dir, "masks", s))
    )
    size = sum(
        os.path.getsize(os.path.join(dp, f))
        for s in ["train", "val", "test"]
        for dp, _, fns in os.walk(os.path.join(args.data_dir, "masks", s))
        for f in fns
    )
    print(f"\nDone. {total} masks, {size/1e9:.1f} GB")
    print(f"Masks saved to: {args.data_dir}/masks/")


if __name__ == "__main__":
    main()
