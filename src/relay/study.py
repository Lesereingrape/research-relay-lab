"""The study: one frozen policy, six harnesses, a budget sweep, three seeds.

`run_seed` trains on oracle traces from one ledger and evaluates on queries drawn
from a *different* ledger, so a policy that memorised a training corpus cannot carry
it into the held-out set -- the unit of replication is the seed, and the seed changes
both initialisation and data.

Every number the README prints is a mean over held-out queries, and the raw
per-seed values ship next to it, because a gap carried by one lucky seed is not a
result.
"""

from __future__ import annotations

import random
import statistics

from relay.harness import (
    ARMS,
    BUDGETS,
    QUERY_ROLE,
    batch,
    chain_records,
    encode,
    lines_payload,
    look,
    run_many,
)
from relay.kb import MIN_DEPTH, Example, make_dataset
from relay.metrics import hop_profile, score
from relay.model import build
from relay.train import STEPS, build_traces, train

SEEDS = (0, 1, 2)
N_TRAIN = 4000
N_EVAL = 1200
EVAL_DEPTHS = (MIN_DEPTH, MIN_DEPTH + 1)
# The arms whose researcher does the reading under test: `oracle_relay` is handed the
# right value, so a misread profile for it would be a straight line at zero.
HOP_ARMS = ("typed_relay", "cited_relay", "full_context")


def teacher_forced(model, exs: list[Example]) -> dict[str, float]:
    """What the policy gets right when the harness does not interfere.

    One row per hop, evidence handed to it, and the question is only: which value do
    you write, and where do you look next. Anything the arms lose on top of this is
    the relay's doing, not the model's.
    """
    rows, note_t, nxt_t = [], [], []
    for ex in exs:
        recs = chain_records(ex)
        for i in range(ex.depth):
            payload = lines_payload(look(ex, recs[i], i))
            rows.append(encode(QUERY_ROLE, recs[i], ex, payload))
            note_t.append(ex.chain[i])
            nxt_t.append(ex.corpus.fresh(recs[i]).nxt)
    note_hit = nxt_hit = 0
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        out = batch(model, chunk)
        pred_note = out.note.argmax(-1).tolist()
        pred_nxt = out.nxt.argmax(-1).tolist()
        note_hit += sum(p == t for p, t in
                        zip(pred_note, note_t[start:start + len(chunk)], strict=True))
        nxt_hit += sum(p == t for p, t in
                       zip(pred_nxt, nxt_t[start:start + len(chunk)], strict=True))
    n = max(1, len(rows))
    return {"note_acc": round(note_hit / n, 4), "next_acc": round(nxt_hit / n, 4)}


def _subset(exs: list[Example], traces, depth: int):
    keep = [(ex, tr) for ex, tr in zip(exs, traces, strict=True) if ex.depth == depth]
    return keep


def run_seed(seed: int) -> dict:
    rng = random.Random(seed)
    train_exs = make_dataset(N_TRAIN, seed, EVAL_DEPTHS)
    eval_exs = make_dataset(N_EVAL, 10_000 + seed, EVAL_DEPTHS)
    model = build(seed)
    curve = train(model, build_traces(train_exs, rng), seed, steps=STEPS)

    arms: dict[str, dict] = {}
    per_depth: dict[str, dict] = {}
    budget: dict[str, dict] = {}
    hops: dict[str, list] = {}
    for arm in ARMS:
        traces = run_many(model, eval_exs, arm)
        pairs = list(zip(eval_exs, traces, strict=True))
        arms[arm] = score(pairs)
        per_depth[arm] = {str(d): score(_subset(eval_exs, traces, d))
                          for d in EVAL_DEPTHS}
        budget[arm] = {}
        for b in BUDGETS[arm]:
            budget[arm][str(b)] = score(list(zip(
                eval_exs, run_many(model, eval_exs, arm, b), strict=True)))
        if arm in HOP_ARMS:
            hops[arm] = hop_profile(pairs)

    return {
        "seed": seed,
        "loss_curve": curve,
        "steps": STEPS,
        "policy": teacher_forced(model, eval_exs),
        "arms": arms,
        "per_depth": per_depth,
        "budget": budget,
        "hops": hops,
    }


def _stats(values: list) -> dict:
    """Mean and spread over the seeds that could answer at all.

    A field no seed measured is ``None`` here, not 0.0 -- see `metrics.score` for
    which arm cannot be asked which question.
    """
    seen = [v for v in values if v is not None]
    if not seen:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.fmean(seen), 4),
            "std": round(statistics.pstdev(seen), 4)}


def _mean_profile(profiles: list[list]) -> list:
    """Elementwise mean over per-seed hop profiles; `None` where no seed reached a hop."""
    length = max((len(p) for p in profiles), default=0)
    return [_stats([p[i] for p in profiles if i < len(p)])["mean"]
            for i in range(length)]


def aggregate(per_seed: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for arm in ARMS:
        block = {name: _stats([ps["arms"][arm][name] for ps in per_seed])
                 for name in per_seed[0]["arms"][arm]}
        block["per_depth"] = {
            str(d): {name: _stats([ps["per_depth"][arm][str(d)][name] for ps in per_seed])
                     for name in per_seed[0]["per_depth"][arm][str(d)]}
            for d in EVAL_DEPTHS}
        block["budget"] = {
            str(b): {name: _stats([ps["budget"][arm][str(b)][name] for ps in per_seed])
                     for name in per_seed[0]["budget"][arm][str(b)]}
            for b in BUDGETS[arm]}
        block["hops"] = (_mean_profile([ps["hops"][arm] for ps in per_seed])
                         if arm in per_seed[0]["hops"] else None)
        out[arm] = block
    out["_policy"] = {name: _stats([ps["policy"][name] for ps in per_seed])
                      for name in per_seed[0]["policy"]}
    return out
