#!/usr/bin/env python3
"""
GPT-2 text generation from a trained checkpoint.

Usage:
    python gpt2_generate.py --model ../models/gpt2_ddp_best.pth --prompt "The brain"
    python gpt2_generate.py --model ../models/gpt2_ddp_best.pth --interactive
"""
import argparse
import math
import torch
import torch.nn as nn
import tiktoken

from gpt2_model import GPT2


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
