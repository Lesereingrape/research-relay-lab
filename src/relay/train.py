"""Behaviour cloning on oracle traces: one policy, both roles.

The teacher is `relay.kb`: it walks the true chain, writes the value it stood on,
and hands the reporter a notebook that is *full*. That is exactly the training
distribution a real research pipeline would have -- nobody annotates the runs where
the sub-agent was handed half the evidence -- and it is why the reporter's behaviour
under a clipped notebook has to be measured rather than assumed.

Nothing here learns a search policy from reward: the point of the lab is where the
evidence goes, not whether the agent can be coaxed into finding it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from relay.harness import (
    HEAD_PREFIX,
    MAX_PAYLOAD,
    QUERY_ROLE,
    REPORT_ROLE,
    encode,
    look,
    notebook,
)
from relay.kb import Example
from relay.model import Relay

BATCH = 128
LR = 0.005
STEPS = 2500
EVAL_EVERY = 250


@dataclass(frozen=True)
class Row:
    ids: tuple[int, ...]
    note: int
    nxt: int
    ans: int
    cite: int


def _researcher_rows(ex: Example) -> list[Row]:
    """One row per hop: the hits a search returned, and what to take from them."""
    recs = list(ex.records)
    return [Row(tuple(encode(QUERY_ROLE, recs[i], ex,
                             [t for line in look(ex, recs[i], i)
                              for t in line.tokens()])),
               note=ex.chain[i], nxt=ex.corpus.fresh(recs[i]).nxt, ans=-1, cite=-1)
           for i in range(ex.depth)]


def _reporter_rows(ex: Example, capacity: int, cost: int) -> list[Row]:
    """The fold, taught in both shapes the arms will ask for it.

    A notebook of typed values, and -- for the arm that skips the notebook and reads
    the hits -- the raw search results themselves. Without the second kind of row the
    "no notebook" arm would be compared while untrained, and any gap it showed would
    belong to the training set rather than to the harness.
    """
    recs = list(ex.records)
    rows = []
    for payload in (
            notebook(list(ex.chain[:capacity])
                     + [None] * max(0, capacity - ex.depth), recs, cost)[:MAX_PAYLOAD],
            [t for i in range(ex.depth) for line in look(ex, recs[i], i)
             for t in line.tokens()][:(MAX_PAYLOAD - HEAD_PREFIX)]):
        rows.append(Row(tuple(encode(REPORT_ROLE, ex.start, ex, payload)),
                        note=-1, nxt=-1, ans=ex.answer, cite=ex.depth))
    return rows


def build_traces(exs: list[Example], rng: random.Random) -> list[Row]:
    """One researcher row per hop plus one report row per query, over full notebooks.

    Capacities above the chain length are included so the reporter learns that a
    blank slot is not a zero; capacities below it are *not*, which is the whole
    asymmetry the study then measures.
    """
    rows: list[Row] = []
    costs = (1, 2)
    for ex in exs:
        cost = costs[rng.randrange(len(costs))]
        rows.extend(_researcher_rows(ex))
        rows.extend(_reporter_rows(ex, ex.depth + rng.randrange(0, 3), cost))
    return rows


def _stack(rows: list[Row], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(r.ids) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
    mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        ids[i, :len(row.ids)] = torch.tensor(row.ids, dtype=torch.long, device=device)
        mask[i, :len(row.ids)] = True
    return ids, mask


def _labels(rows: list[Row], attr: str, device: torch.device) -> torch.Tensor | None:
    values = [getattr(r, attr) for r in rows]
    if all(v < 0 for v in values):
        return None
    return torch.tensor(values, dtype=torch.long, device=device)


def _loss(model: Relay, rows: list[Row], device: torch.device) -> torch.Tensor:
    ids, mask = _stack(rows, device)
    out = model(ids, mask)
    total = torch.zeros((), device=device)
    for head, attr in (("note", "note"), ("nxt", "nxt"), ("ans", "ans"),
                       ("cite", "cite")):
        target = _labels(rows, attr, device)
        if target is None:
            continue
        total = total + nn.functional.cross_entropy(getattr(out, head), target)
    return total


def train(model: Relay, rows: list[Row], seed: int, steps: int = STEPS,
          batch: int = BATCH, lr: float = LR) -> list[float]:
    """Split researcher rows from reporter rows, then one Adam pass over both."""
    device = torch.device("cpu")
    rng = random.Random(seed)
    q_rows = [r for r in rows if r.note >= 0]
    r_rows = [r for r in rows if r.ans >= 0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    curve: list[float] = []
    for step in range(steps):
        model.train()
        q = [q_rows[i] for i in rng.choices(range(len(q_rows)), k=batch)]
        r = [r_rows[i] for i in rng.choices(range(len(r_rows)), k=batch)]
        loss = _loss(model, q, device) + _loss(model, r, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % EVAL_EVERY == 0:
            curve.append(round(float(loss.item()) / 2, 4))
    model.eval()
    return curve
