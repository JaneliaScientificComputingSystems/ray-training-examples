#!/usr/bin/env python3
"""
GPT-2 small (117M) model architecture.
Shared by training, evaluation, and generation scripts.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


GPT2_CONFIG = {
    "vocab_size": 50257, "block_size": 1024,
    "n_layer": 12, "n_head": 12, "n_embd": 768,
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=True)
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=True)
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["n_embd"]
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
        x = x + self.mlp(self.ln_2(x))
        return x


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
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-1)
        return logits, loss
