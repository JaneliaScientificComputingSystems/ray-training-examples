#!/usr/bin/env python3
"""
GPT-2 perplexity evaluation on OpenWebText val set (parquet shards).

Usage:
    python gpt2_eval.py --model ../models/gpt2_ddp_best.pth
    python gpt2_eval.py --model ../models/gpt2_ddp_best.pth --num-batches 200
"""
import argparse
import glob
import math
import os
import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq

from gpt2_model import GPT2


# ---------------------------------------------------------------------------
# Val dataset from parquet
# ---------------------------------------------------------------------------

class ValDataset(torch.utils.data.Dataset):
    """Reads tokenized sequences from parquet val shards."""
    def __init__(self, data_dir, seq_len=1024):
        val_dir = os.path.join(data_dir, "val")
        files = sorted(glob.glob(os.path.join(val_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {val_dir}")
        rows = []
        for f in files:
            table = pq.read_table(f, columns=["input_ids"])
            for batch in table.to_batches():
                for row in batch.column("input_ids"):
                    tokens = row.as_py()
                    if len(tokens) >= seq_len + 1:
                        rows.append(tokens[:seq_len + 1])
        self.data = np.array(rows, dtype=np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:self.seq_len], dtype=torch.long)
        y = torch.tensor(tokens[1:self.seq_len + 1], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT2()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_loss = ckpt.get("val_loss", float("nan"))
    iter_num = ckpt.get("iter_num", "?")
    print(f"Loaded: {path}")
    print(f"  Iter: {iter_num}  |  Saved val loss: {val_loss:.4f}  "
          f"(ppl {math.exp(val_loss):.1f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data-dir", default="/nrs/ml_datasets/openwebtext")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Limit eval batches (default: full val set)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    model = load_model(args.model, device)
    dataset = ValDataset(args.data_dir)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_amp = device.type == "cuda"

    losses = []
    total_tokens = 0
    label = "full val set" if args.num_batches is None \
            else f"{args.num_batches} batches"
    print(f"\nEvaluating ({label})...")

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if args.num_batches and i >= args.num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            losses.append(loss.item())
            total_tokens += x.numel()
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1:4d} | running ppl "
                      f"{math.exp(sum(losses)/len(losses)):.2f}")

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)
    bpc = mean_loss / math.log(2) / 4.0

    print(f"\n{'='*50}")
    print(f"Val loss:    {mean_loss:.4f}")
    print(f"Perplexity:  {ppl:.2f}")
    print(f"Bits/char:   {bpc:.3f}  (estimate, ~4 chars/token)")
    print(f"Tokens eval: {total_tokens:,}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
