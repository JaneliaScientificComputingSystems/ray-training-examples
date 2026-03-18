#!/usr/bin/env python3
"""
GPT-2 text generation from a trained checkpoint.

Usage:
    python gpt2_generate.py --model ./models/gpt2_ddp_best.pth --prompt "The brain"
    python gpt2_generate.py --model ./models/gpt2_ddp_best.pth --interactive
"""
import argparse
import math
import torch
import torch.nn as nn
import tiktoken

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
        self.gelu = nn.GELU()
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
        self.cfg = cfg
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
# Generation
# ---------------------------------------------------------------------------

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT2()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    val_loss = ckpt.get("val_loss", float("nan"))
    iter_num = ckpt.get("iter_num", "?")
    print(f"Loaded: {path}")
    print(f"  Iter: {iter_num}  |  Val loss: {val_loss:.4f}  "
          f"(ppl {math.exp(val_loss):.1f})")
    return model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=200):
    block_size = model.cfg["block_size"]
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits.float(), dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)
    return idx


def run_prompt(model, enc, prompt, args, device):
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(model, x, args.max_tokens, args.temperature, args.top_k)
    text = enc.decode(out[0].tolist())
    print(f"\n{text}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--prompt", default="The",
                        help="Text prompt to continue")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    model = load_model(args.model, device)
    enc = tiktoken.get_encoding("gpt2")

    if args.interactive:
        print(f"\nInteractive mode (temp={args.temperature}, "
              f"top_k={args.top_k}, max_tokens={args.max_tokens})")
        print("Type a prompt, Ctrl+C to exit.\n")
        try:
            while True:
                prompt = input("Prompt> ").strip()
                if prompt:
                    run_prompt(model, enc, prompt, args, device)
        except (KeyboardInterrupt, EOFError):
            print()
    else:
        run_prompt(model, enc, args.prompt, args, device)


if __name__ == "__main__":
    main()
