"""One frozen policy, six harnesses around it.

The trunk is a small decoder-free transformer: a sum-pooled encoder over a token
sequence, with four typed heads. Nothing here generates text -- every head emits a
class over a closed vocabulary, so a rollout is a handful of matrix multiplies and
the whole study runs on a laptop CPU.

The point of the shape is that the *model is constant across arms*. The harness
decides what the policy is allowed to see and what it is allowed to keep, so any
difference between the rows of the results table is a difference in the interface,
not in the weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from relay.kb import N_CITE, N_REC, N_TOK, N_VAL

MAXLEN = 48
D_MODEL = 72
HEADS = 4
LAYERS = 3
FF = 144
DROPOUT = 0.0


@dataclass(frozen=True)
class Logits:
    note: torch.Tensor   # (B, N_VAL)  what value to write for this hop
    nxt: torch.Tensor    # (B, N_REC)  which record to search next
    ans: torch.Tensor    # (B, N_VAL)  the reported answer
    cite: torch.Tensor   # (B, N_CITE) how many hops the report claims


class Relay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_TOK, D_MODEL)
        self.pos = nn.Embedding(MAXLEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=HEADS, dim_feedforward=FF,
            dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.note = nn.Linear(D_MODEL, N_VAL)
        self.nxt = nn.Linear(D_MODEL, N_REC)
        self.ans = nn.Linear(D_MODEL, N_VAL)
        self.cite = nn.Linear(D_MODEL, N_CITE)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> Logits:
        """`ids` (B, L) token ids, `mask` (B, L) True where real."""
        length = ids.shape[1]
        pos = torch.arange(length, device=ids.device).clamp(max=MAXLEN - 1)
        x = self.embed(ids) + self.pos(pos)
        x = self.trunk(x, src_key_padding_mask=~mask)
        x = self.norm(x) * mask.unsqueeze(-1)
        # Sum, not mean, over the real positions: the reporter has to fold the values
        # it was handed, and a mean would let the *number of notes* leak into the
        # magnitude of every one of them. Padding positions contribute nothing.
        pooled = x.sum(1)
        return Logits(note=self.note(pooled), nxt=self.nxt(pooled),
                      ans=self.ans(pooled), cite=self.cite(pooled))


def build(seed: int) -> Relay:
    torch.manual_seed(seed)
    model = Relay()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(D_MODEL))
    return model


def count_parameters() -> int:
    return sum(p.numel() for p in Relay().parameters())
