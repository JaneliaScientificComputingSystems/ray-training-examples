#!/usr/bin/env python3
"""
GPT-2 text generation — interactive and batch modes.
Loads any checkpoint from the training script.

Usage:
    python gpt2_generate.py --model ./models/gpt2_ddp_best_YYYYMMDD.pth
    python gpt2_generate.py --model ./models/gpt2_ddp_best_YYYYMMDD.pth \
        --prompt "The neural network was trained on" --max-tokens 200
    python gpt2_generate.py --model ./models/gpt2_ddp_best_YYYYMMDD.pth \
        --prompts-file prompts.txt
"""
import argparse
import os
import math
import torch
import tiktoken

import importlib.util
spec = importlib.util.spec_from_file_location(
    "gpt2_train", os.path.join(os.path.dirname(__file__),
                               "gpt2_distributed_training.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
GPT2 = mod.GPT2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--prompt",      default="The",
                        help="Text prompt to continue")
    parser.add_argument("--prompts-file", default=None,
                        help="File with one prompt per line — runs all")
    parser.add_argument("--max-tokens",  type=int, default=200,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy, 1=full random)")
    parser.add_argument("--top-k",       type=int, default=200,
                        help="Top-k sampling — 0 to disable")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of independent samples per prompt")
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive REPL mode")
    return parser.parse_args()


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt  = torch.load(path, map_location=device)
    model = GPT2().to(device)
    state = {k.replace("module.", ""): v
             for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    val_loss = ckpt.get("val_loss", float("nan"))
    iter_num = ckpt.get("iter_num", "?")
    print(f"Model: {path}")
    print(f"  iter {iter_num} | val_loss {val_loss:.4f} "
          f"| ppl {math.exp(val_loss):.1f}")
    return model


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0,
             top_k=None, device=None):
    """Autoregressive generation with temperature and top-k sampling."""
    block_size = GPT2.CFG["block_size"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx   = torch.amp.autocast(device_type="cuda", dtype=dtype) \
            if (device and device.type == "cuda") else torch.no_grad()

    for _ in range(max_new_tokens):
        # Crop context to block size
        idx_cond = idx if idx.size(1) <= block_size \
                   else idx[:, -block_size:]
        with ctx:
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits.float(), dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tok), dim=1)

    return idx


def run_prompt(model, enc, prompt, max_tokens, temperature,
               top_k, num_samples, device):
    ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    x   = x.repeat(num_samples, 1)

    print(f"\nPrompt: {prompt!r}")
    print("-" * 60)

    out = generate(model, x, max_tokens, temperature, top_k, device)

    for i in range(num_samples):
        tokens  = out[i].tolist()
        text    = enc.decode(tokens)
        if num_samples > 1:
            print(f"--- Sample {i+1} ---")
        print(text)
        print()


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args.model, device)
    enc   = tiktoken.get_encoding("gpt2")

    gen_kwargs = dict(
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
        top_k       = args.top_k if args.top_k > 0 else None,
        num_samples = args.num_samples,
        device      = device,
    )

    if args.interactive:
        print("\nInteractive mode — enter prompts, Ctrl+C to exit")
        print(f"  temperature={args.temperature}  "
              f"top_k={args.top_k}  max_tokens={args.max_tokens}")
        try:
            while True:
                prompt = input("\nPrompt> ").strip()
                if prompt:
                    run_prompt(model, enc, prompt, **gen_kwargs)
        except KeyboardInterrupt:
            print("\nExiting.")

    elif args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
        print(f"Running {len(prompts)} prompts from {args.prompts_file}")
        for prompt in prompts:
            run_prompt(model, enc, prompt, **gen_kwargs)

    else:
        run_prompt(model, enc, args.prompt, **gen_kwargs)


if __name__ == "__main__":
    main()
