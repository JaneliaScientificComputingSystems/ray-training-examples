#!/usr/bin/env python3
"""
GPT-2 small (117M) distributed training on OpenWebText.
Supports DDP and FSDP — toggle with --mode flag.
Works on all Janelia GPU queues — IB and Ethernet NCCL.
"""
import argparse
import contextlib
import os
import math
import time
import numpy as np
import ray
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus",    type=int, required=True)
    parser.add_argument("--num-nodes",   type=int, required=True)
    parser.add_argument("--mode",        choices=["ddp", "fsdp"], default="ddp",
                        help="DDP: replicated params. FSDP: sharded params (use for larger models)")
    parser.add_argument("--batch-size",  type=int, default=8,
                        help="micro-batch per GPU (tokens = batch_size * seq_len)")
    parser.add_argument("--seq-len",     type=int, default=1024)
    parser.add_argument("--grad-accum",  type=int, default=8,
                        help="gradient accumulation steps — effective batch = batch_size * grad_accum * num_gpus")
    parser.add_argument("--max-iters",   type=int, default=5000,
                        help="training iterations (5000 ~ 3-4h on 8x H200)")
    parser.add_argument("--eval-iters",  type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--lr",          type=float, default=6e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--data-dir",    type=str,
                        default=os.path.expanduser("~/datasets/openwebtext"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPT-2 model — minimal clean implementation
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.c_attn  = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=True)
        self.c_proj  = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=True)
        self.n_head  = cfg["n_head"]
        self.n_embd  = cfg["n_embd"]
        self.register_buffer("bias",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att.float(), dim=-1).to(x.dtype)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=True)
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=True)
        self.act    = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg["n_embd"])
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """GPT-2 small: 117M params, 12 layers, 12 heads, 768 embed dim."""
    CFG = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
               vocab_size=50304, bias=True)  # vocab padded to nearest 64

    def __init__(self):
        super().__init__()
        cfg = self.CFG
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg["vocab_size"], cfg["n_embd"]),
            wpe  = nn.Embedding(cfg["block_size"], cfg["n_embd"]),
            drop = nn.Dropout(0.0),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])]),
            ln_f = nn.LayerNorm(cfg["n_embd"]),
        ))
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.drop(
                   self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x    = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data    = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx:idx + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# Learning rate schedule — cosine with warmup
# ---------------------------------------------------------------------------

def get_lr(it, warmup_iters, max_iters, lr, min_lr=6e-5):
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * decay))


# ---------------------------------------------------------------------------
# IB verification
# ---------------------------------------------------------------------------

def verify_ib_in_use():
    import subprocess
    ib_devs = ["mlx5_0","mlx5_1","mlx5_2","mlx5_3",
                "mlx5_4","mlx5_6","mlx5_7","mlx5_8"]
    print("--- IB device check ---")
    for dev in ib_devs:
        try:
            r = subprocess.run(["ibv_devinfo", "-d", dev],
                               capture_output=True, text=True, timeout=5)
            link  = next((l.strip() for l in r.stdout.splitlines()
                          if "link_layer" in l), "unknown")
            ok = "InfiniBand" in link
            print(f"  {'OK  ' if ok else 'WARN'} {dev}: {link}")
        except Exception as e:
            print(f"  ERR  {dev}: {e}")
    if "mlx5_5" in os.environ.get("NCCL_IB_HCA", ""):
        print("  ERROR: mlx5_5 (Ethernet) in NCCL_IB_HCA — remove it!")
    else:
        print("  OK   mlx5_5 (Ethernet) excluded")
    print("-----------------------")


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, iter_num, val_loss, config,
                    is_best=False, world_rank=0):
    if world_rank != 0:
        return None
    model_dir = config["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "best" if is_best else f"iter{iter_num}"
    path   = os.path.join(model_dir, f"gpt2_{config['mode']}_{suffix}_{ts}.pth")
    raw    = model.module if hasattr(model, "module") else model
    torch.save({
        "iter_num":        iter_num,
        "model_state":     raw.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss":        val_loss,
        "mode":            config["mode"],
    }, path)
    print(f"Checkpoint: {path}  (val_loss {val_loss:.4f})")
    return path


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    raw  = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed iter {ckpt['iter_num']}, val_loss {ckpt['val_loss']:.4f}")
    return ckpt["iter_num"] + 1, ckpt["val_loss"]


# ---------------------------------------------------------------------------
# Training function (runs on each Ray worker)
# ---------------------------------------------------------------------------

def train_func(config):
    import ray.train
    import torch.distributed as dist

    local_rank  = ray.train.get_context().get_local_rank()
    world_rank  = ray.train.get_context().get_world_rank()
    world_size  = ray.train.get_context().get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Use bf16 on H100/H200 (Ampere+), fp16 otherwise
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx    = torch.amp.autocast(device_type="cuda", dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    if world_rank == 0:
        nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
        backend = "InfiniBand (GPUDirect RDMA)" if nccl_ib == "0" else "Ethernet"
        print(f"Workers: {world_size} | GPU: {torch.cuda.get_device_name(device)}")
        print(f"Mode: {config['mode'].upper()} | Network: {backend} | dtype: {dtype}")
        print(f"Effective batch: {config['batch_size'] * config['grad_accum'] * world_size}"
              f" sequences ({config['batch_size'] * config['grad_accum'] * world_size * config['seq_len']:,} tokens)")
        if nccl_ib == "0":
            verify_ib_in_use()

    # Data
    train_ds = TokenDataset(
        os.path.join(config["data_dir"], "train.bin"), config["seq_len"])
    val_ds   = TokenDataset(
        os.path.join(config["data_dir"], "val.bin"),   config["seq_len"])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                        rank=world_rank, shuffle=True)
    train_loader  = DataLoader(train_ds, batch_size=config["batch_size"],
                               sampler=train_sampler, num_workers=4,
                               pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(val_ds, batch_size=config["batch_size"],
                               num_workers=2, pin_memory=True)

    # Model
    model = GPT2().to(device)

    if world_rank == 0:
        print(f"GPT-2 parameters: {model.num_params()/1e6:.1f}M")

    if config["mode"] == "fsdp":
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            device_id=device,
            # AllGather happens over IB before each forward pass per shard
        )
    else:
        model = DDP(model, device_ids=[local_rank])
        # AllReduce happens over IB after backward pass

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"],
        betas=(0.9, 0.95), weight_decay=0.1)

    start_iter = 0
    best_val   = float("inf")

    if config.get("resume_checkpoint") and world_rank == 0:
        start_iter, best_val = load_checkpoint(
            config["resume_checkpoint"], model, optimizer, device)

    if world_size > 1:
        si = torch.tensor(start_iter, device=device)
        bv = torch.tensor(best_val,   device=device)
        dist.broadcast(si, src=0)
        dist.broadcast(bv, src=0)
        start_iter = int(si.item())
        best_val   = bv.item()

    # Training loop
    train_iter   = iter(train_loader)
    t0           = time.time()
    tokens_total = 0

    for it in range(start_iter, config["max_iters"]):
        # LR schedule
        lr = get_lr(it, config["warmup_iters"], config["max_iters"], config["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(config["grad_accum"]):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_sampler.set_epoch(it)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Only sync gradients on the last accumulation step
            sync = (micro == config["grad_accum"] - 1)
            ctx_sync = model.no_sync() if not sync and hasattr(model, "no_sync") \
                       else contextlib.nullcontext()

            with ctx_sync:
                with ctx:
                    _, loss = model(x, y)
                    loss    = loss / config["grad_accum"]
                scaler.scale(loss).backward()
                accum_loss += loss.item()

        tokens_total += (config["batch_size"] * config["grad_accum"]
                         * config["seq_len"] * world_size)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if it % 50 == 0 and world_rank == 0:
            dt       = time.time() - t0
            tok_sec  = tokens_total / dt
            print(f"iter {it:5d} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {tok_sec/1e3:.1f}K tok/s | "
                  f"{dt:.0f}s elapsed")

        # Eval
        if it % config["eval_interval"] == 0 and it > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi, (vx, vy) in enumerate(val_loader):
                    if vi >= config["eval_iters"]:
                        break
                    vx = vx.to(device, non_blocking=True)
                    vy = vy.to(device, non_blocking=True)
                    with ctx:
                        _, vloss = model(vx, vy)
                    val_losses.append(vloss.item())
            val_loss = float(np.mean(val_losses))
            model.train()

            if world_rank == 0:
                tok_sec = tokens_total / (time.time() - t0)
                print(f"  val_loss {val_loss:.4f} | "
                      f"throughput {tok_sec/1e3:.1f}K tok/s")

            if config.get("save_models") and val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, it, val_loss,
                                config, is_best=True,
                                world_rank=world_rank)

            ray.train.report({
                "iter":     it,
                "val_loss": val_loss,
                "tok_sec":  tokens_total / (time.time() - t0),
                "best_val": best_val,
                "mode":     config["mode"],
            })

    # Final checkpoint
    if config.get("save_models"):
        save_checkpoint(model, optimizer, config["max_iters"],
                        best_val, config, is_best=False,
                        world_rank=world_rank)

    if world_rank == 0:
        elapsed  = time.time() - t0
        tok_sec  = tokens_total / elapsed
        print(f"\nDone. {tokens_total/1e9:.2f}B tokens | "
              f"{tok_sec/1e3:.1f}K tok/s avg | {elapsed/3600:.1f}h")
        print(f"Best val loss: {best_val:.4f}")


def main():
    args = parse_args()

    train_bin = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_bin):
        print(f"ERROR: {train_bin} not found — run the dataset prep step first")
        return

    model_dir = os.path.join(os.path.abspath(os.getcwd()), "models")

    ray.init(address="auto")
    nccl_ib = os.environ.get("NCCL_IB_DISABLE", "1")
    backend = "InfiniBand" if nccl_ib == "0" else "Ethernet"
    print(f"Ray: {ray.available_resources()}")
    print(f"Mode: {args.mode.upper()} | Network: {backend}")
    print(f"Max iters: {args.max_iters} | "
          f"Effective batch: {args.batch_size * args.grad_accum * args.num_gpus} seqs "
          f"({args.batch_size * args.grad_accum * args.num_gpus * args.seq_len:,} tokens)")

    scaling_config = ScalingConfig(
        num_workers=args.num_gpus, use_gpu=True,
        resources_per_worker={"CPU": 7, "GPU": 1})
    run_config = RunConfig(
        name=f"gpt2_{args.mode}",
        storage_path=f"/scratch/{os.getenv('USER','unknown')}/ray_results")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "mode":              args.mode,
            "batch_size":        args.batch_size,
            "seq_len":           args.seq_len,
            "grad_accum":        args.grad_accum,
            "max_iters":         args.max_iters,
            "eval_iters":        args.eval_iters,
            "eval_interval":     args.eval_interval,
            "lr":                args.lr,
            "warmup_iters":      args.warmup_iters,
            "num_nodes":         args.num_nodes,
            "save_models":       args.save_models,
            "resume_checkpoint": args.resume,
            "data_dir":          args.data_dir,
            "model_dir":         model_dir,
        },
        scaling_config=scaling_config,
        run_config=run_config)

    result = trainer.fit()
    if result and result.metrics:
        print(f"Best val loss: {result.metrics.get('best_val', 'N/A'):.4f}")
    ray.shutdown()


if __name__ == "__main__":
    main()
