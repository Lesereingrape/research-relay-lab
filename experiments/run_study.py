"""Run the relay comparison and write results/frontier.json.

Usage:  PYTHONPATH=src python experiments/run_study.py [--out PATH]

The artifact records the environment it came off and every individual seed, so a
reader can see how much of a gap is one lucky initialisation. Float reduction
order over a batch follows the thread count and the torch build, so a rerun is
bit-exact *in that environment* and merely close in another -- ``runtime_sec`` is
the one field a rerun is allowed to move.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relay.harness import (
    ARMS,
    BUDGETS,
    COST,
    DEFAULT_BUDGET,
    DISTRACTORS,
    MAX_PAYLOAD,
)
from relay.kb import MAX_DEPTH, MIN_DEPTH, N_REC, N_TOK, N_VAL, VERSIONS
from relay.model import count_parameters
from relay.study import EVAL_DEPTHS, N_EVAL, N_TRAIN, SEEDS, aggregate, run_seed
from relay.train import BATCH, LR, STEPS

HEADLINE = ("answer_acc", "walked_ok", "wrong_value_rate", "lost_rate",
            "carried_rate", "cite_overclaim", "partial_sum_rate",
            "acc_full_notebook", "reporter_tokens")


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def _fmt(value) -> str:
    """`n/a` where an arm cannot be asked the question (see `relay.metrics.score`)."""
    return "n/a" if value is None else f"{value:.3f}"


def main(out: str | None = None, against: str | None = None) -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    arms = aggregate(per_seed)
    artifact = {
        "config": {
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "eval_depths": list(EVAL_DEPTHS),
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
            "steps": STEPS,
            "batch": BATCH,
            "lr": LR,
            "params": count_parameters(),
            "cost": COST,
            "default_budget": DEFAULT_BUDGET,
            "budgets": {arm: list(BUDGETS[arm]) for arm in ARMS},
            "distractors": DISTRACTORS,
            "n_tokens": N_TOK,
            "n_records": N_REC,
            "n_values": N_VAL,
            "versions": list(VERSIONS),
            "max_payload": MAX_PAYLOAD,
            "chance": round(1.0 / N_VAL, 4),
        },
        "arms": arms,
        "per_seed": {str(ps["seed"]): {arm: {name: ps["arms"][arm][name]
                                            for name in HEADLINE}
                                       for arm in ARMS}
                     for ps in per_seed},
        "per_seed_policy": {str(ps["seed"]): ps["policy"] for ps in per_seed},
        "environment": environment(),
        "runtime_sec": round(time.time() - t0, 1),
    }
    dest = Path(out or "results/frontier.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote {dest} in {artifact['runtime_sec']}s")
    for arm in ARMS:
        a = arms[arm]
        print(f"{arm:13s} acc {_fmt(a['answer_acc']['mean'])}"
              f" +/- {_fmt(a['answer_acc']['std'])}"
              f"  walked {_fmt(a['walked_ok']['mean'])}"
              f"  lost {_fmt(a['lost_rate']['mean'])}"
              f"  overclaim {_fmt(a['cite_overclaim']['mean'])}"
              f"  tok {_fmt(a['reporter_tokens']['mean'])}")
    print("policy:", {k: _fmt(v["mean"]) for k, v in arms["_policy"].items()})
    if against:
        _compare(artifact, Path(against), dest)


def _compare(fresh: dict, committed: Path, written: Path) -> None:
    """Fail if a rerun moved anything other than the wall-clock."""
    if committed.resolve() == written.resolve():
        raise SystemExit("--against has to name the committed artifact while --out "
                         "names a scratch copy; pointing both at one file compares a "
                         "run with itself\n")
    other = json.loads(committed.read_text(encoding="utf-8"))
    a, b = {**fresh, "runtime_sec": None}, {**other, "runtime_sec": None}
    if a == b:
        print(f"identical to {committed} in every field but runtime_sec")
        return
    moved = [k for k in a if a[k] != b.get(k)]
    raise SystemExit(f"rerun differs from {committed} in: {', '.join(moved)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_study")
    parser.add_argument("--out", default=None,
                        help="where to write the artifact (default: results/frontier.json)")
    parser.add_argument("--against", default=None,
                        help="a committed artifact to compare this run with, field for "
                             "field; only runtime_sec is allowed to move")
    args = parser.parse_args()
    main(args.out, args.against)
