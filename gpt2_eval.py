#!/usr/bin/env python3
"""
GPT-2 perplexity evaluation on OpenWebText val set.
Runs on a single GPU — no distributed setup needed.
Reports per-token perplexity and bits-per-character.

Usage:
    python gpt2_eval.py --model ./models/gpt2_ddp_best_YYYYMMDD.pth
    python gpt2_eval.py --model ./models/gpt2_fsdp_best_YYYYMMDD.pth --num-batches 500
"""
import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Re-use model definition from training script
import importlib.util, sys
spec = importlib.util.spec_from_file_location(
    "gpt2_train", os.path.join(os.path.dirname(__file__),
                               "gpt2_distributed_training.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GPT2        = mod.GPT2
TokenDataset = mod.TokenDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--data-dir",   default=os.path.expanduser(
                                            "~/datasets/openwebtext"))
    parser.add_argument("--seq-len",    type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Limit evaluation batches (default: full val set)")
    parser.add_argument("--device",     default="auto")
    return parser.parse_args()


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)

    model = GPT2().to(device)
    # Strip DDP/FSDP wrapper prefix if present
    state = ckpt["model_state"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    mode     = ckpt.get("mode", "unknown")
    val_loss = ckpt.get("val_loss", float("nan"))
    iter_num = ckpt.get("iter_num", "unknown")
    print(f"Loaded: {path}")
    print(f"  Training mode:  {mode.upper()}")
    print(f"  Saved at iter:  {iter_num}")
    print(f"  Saved val loss: {val_loss:.4f}  "
          f"(ppl {math.exp(val_loss):.1f})")
    return model


def evaluate(model, data_dir, seq_len, batch_size, num_batches, device):
    val_bin = os.path.join(data_dir, "val.bin")
    if not os.path.exists(val_bin):
        raise FileNotFoundError(f"val.bin not found at {val_bin}")

    dataset = TokenDataset(val_bin, seq_len)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx   = torch.amp.autocast(device_type="cuda", dtype=dtype) \
            if device.type == "cuda" else torch.no_grad()

    losses      = []
    total_tokens = 0

    print(f"\nEvaluating on val set "
          f"({'full' if num_batches is None else str(num_batches)+' batches'})...")

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if num_batches and i >= num_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                _, loss = model(x, y)
            losses.append(loss.item())
            total_tokens += x.numel()
            if (i + 1) % 50 == 0:
                running_ppl = math.exp(sum(losses) / len(losses))
                print(f"  batch {i+1:4d} | "
                      f"running ppl {running_ppl:.2f}")

    mean_loss = sum(losses) / len(losses)
    ppl       = math.exp(mean_loss)
    # bits per character: loss (nats) / ln(2) / chars_per_token
    # GPT-2 BPE averages ~4 chars/token on English text
    bpc       = mean_loss / math.log(2) / 4.0

    print(f"\n{'='*50}")
    print(f"Val loss:    {mean_loss:.4f}")
    print(f"Perplexity:  {ppl:.2f}")
    print(f"Bits/char:   {bpc:.3f}  (estimate, ~4 chars/token)")
    print(f"Tokens eval: {total_tokens:,}")
    print(f"{'='*50}")
    print(f"\nReference perplexities (OpenWebText val):")
    print(f"  GPT-2 small  (117M) trained to convergence: ~29 ppl")
    print(f"  GPT-2 medium (345M) trained to convergence: ~24 ppl")
    print(f"  Random baseline (50K vocab):               ~50000 ppl")
    return mean_loss, ppl


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args.model, device)
    evaluate(model, args.data_dir, args.seq_len,
             args.batch_size, args.num_batches, device)


if __name__ == "__main__":
    main()
