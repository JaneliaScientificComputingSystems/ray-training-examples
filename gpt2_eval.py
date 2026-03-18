#!/usr/bin/env python3
"""
GPT-2 perplexity evaluation on OpenWebText val set (parquet shards).

Usage:
    python gpt2_eval.py --model ./models/gpt2_ddp_best.pth
    python gpt2_eval.py --model ./models/gpt2_ddp_best.pth --num-batches 200
"""
import argparse
import glob
import math
import os
import numpy as np
import torch
import torch.nn as nn
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Model architecture — must match gpt2_distributed_training.py
# ---------------------------------------------------------------------------

GPT2_CONFIG = {
    "vocab_size": 50257, "block_size": 1024,
    "n_layer": 12, "n_head": 12, "n_embd": 768,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=True)
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=True)
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
        self.register_buffer("bias",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"])
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"])

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or GPT2_CONFIG
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
            wpe=nn.Embedding(cfg["block_size"], cfg["n_embd"]),
            drop=nn.Dropout(0.0),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]),
            ln_f=nn.LayerNorm(cfg["n_embd"]),
        ))
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


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
    parser.add_argument("--data-dir", default="/nrs/scicompsys/Goran/openwebtext")
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
