"""Command line: check the verifier, watch the notebook truncate, run the demo study.

``relay ledger`` needs no model at all -- it is the rule the data is generated
against, which is the point of an exact verifier: a reader can check the ground truth
without trusting any neural network, and can see what a search returns for a record
whose freshest line is buried among stale ones and other records entirely.

``relay notebook`` is the other half of the check: the harness's arithmetic on a
budget, with a flawless researcher supplying it, so ``lost`` and ``overclaim`` mean
something a reader can compute by hand before believing them in a results table.

``relay demo`` trains the one shared policy on a small split and prints the six
harnesses side by side. The README quickstart shows that output. It is not byte-pinned
the way the two commands above are: a demo trains, and a different thread count can
flip a near-tie argmax on a different machine. A test does check that it runs, that
every arm appears, and that the arms come out in the order the study reports them.
"""

from __future__ import annotations

import argparse
import random

from .harness import (
    ARMS,
    BUDGETS,
    COST,
    DEFAULT_BUDGET,
    DISTRACTORS,
    HEAD_PREFIX,
    MAX_PAYLOAD,
    NOTE_ARMS,
    _visible_fresh,
    lines_payload,
    look,
    notebook,
    run_many,
)
from .kb import N_REC, N_VAL, RECS, VALS, dec, make_dataset, make_example
from .metrics import score
from .model import build, count_parameters
from .study import EVAL_DEPTHS, teacher_forced
from .train import build_traces, train

# Small enough to run while reading the README, large enough that the policy has to
# learn the rule rather than the ledgers: at 800 training ledgers it memorises the
# values and the held-out split (a different ledger per query) collapses to a guess.
DEMO_TRAIN = 2500
DEMO_EVAL = 120
DEMO_STEPS = 1200

# The arms whose budget buys accuracy, printed as the frontier in the demo.
FRONTIER = ("typed_relay", "cited_relay", "full_context")


def _ledger(args) -> str:
    """One query, its exact answer, and the evidence a search hands the researcher."""
    ex = make_example(random.Random(args.seed), args.depth)
    out = [f"query: {ex.question}   ledger seed {args.seed}: "
           f"{len(ex.corpus.lines)} lines over {N_REC} records"]
    rec = ex.start
    total = 0
    for hop in range(ex.depth):
        fresh = ex.corpus.fresh(rec)
        hits = look(ex, rec, hop)
        stale = len(ex.corpus.versions(rec)) - 1
        total = (total + fresh.val) % N_VAL
        out.append(f"hop {hop + 1}: {RECS[rec]} has {stale} stale line(s); a search "
                   f"returns {len(hits)} lines, freshest = "
                   f"{RECS[fresh.rec]} {VALS[fresh.val]} -> {RECS[fresh.nxt]}")
        out.append(f"       write {VALS[fresh.val]}, walk to {RECS[fresh.nxt]}, "
                   f"running sum {VALS[total]}")
        rec = fresh.nxt
    out.append(f"answer: {VALS[ex.answer]} "
               f"({' + '.join(VALS[v] for v in ex.chain)} mod {N_VAL})")
    return "\n".join(out)


def _notebook(args) -> str:
    """The harness's bookkeeping on a budget, with a flawless researcher supplying it.

    The budget is the *notebook* window; the report always pays the head as well, so
    the second line is what a reader's context has to hold in total.
    """
    ex = make_example(random.Random(args.seed), args.depth)
    cost = COST[args.arm]
    capacity = args.budget // cost if cost else 0
    records = list(ex.records)
    notes: list[int] = []
    if args.arm == "full_context":
        payload = lines_payload([line for h, rec in enumerate(records)
                                 for line in look(ex, rec, h)])
        payload = payload[:min(args.budget, MAX_PAYLOAD)]
        carried = _visible_fresh(payload, ex, records)
    else:
        notes = list(ex.chain[:capacity])
        payload = notebook(notes, records, cost)[:MAX_PAYLOAD]
        carried = len(notes)
    out = [f"arm {args.arm}: {args.budget} tokens of notebook at {cost} per fact, so "
           f"{carried} of {ex.depth} hops fit",
           f"report payload: {dec(payload) if payload else '(nothing)'}",
           f"head {HEAD_PREFIX} + payload {len(payload)} = "
           f"{HEAD_PREFIX + len(payload)} tokens the reporter reads; "
           f"carried {carried}, lost {ex.depth - carried}"]
    full = carried >= ex.depth
    if args.arm in NOTE_ARMS:
        if full:
            out.append("every hop survived, so the answer is the fold of the notebook: "
                       f"{VALS[ex.answer]}")
        else:
            out.append(f"a report that cites {ex.depth} hops cites more than the "
                       f"interface delivered; folding the {carried} surviving notes "
                       f"gives {VALS[sum(notes) % N_VAL]} against a true "
                       f"{VALS[ex.answer]}")
    elif args.arm == "full_context":
        if full:
            out.append("the window holds every hop's freshest line, and the fold of "
                       f"the chain is {VALS[ex.answer]}")
        else:
            out.append(f"the clip keeps the freshest line of {carried} of {ex.depth} "
                       "hops")
    else:
        out.append("nothing travels between the query and the report")
    return "\n".join(out)


def _demo(args) -> str:
    """One policy, six harnesses, on a split small enough to train in a few minutes."""
    rng = random.Random(args.seed)
    train_exs = make_dataset(args.train, args.seed, EVAL_DEPTHS)
    eval_exs = make_dataset(DEMO_EVAL, 10_000 + args.seed, EVAL_DEPTHS)
    model = build(args.seed)
    train(model, build_traces(train_exs, rng), args.seed, steps=args.steps)
    policy = teacher_forced(model, eval_exs)
    lines = [f"seed {args.seed}: {args.train} train / {DEMO_EVAL} held-out queries, "
             f"{args.steps} steps, {count_parameters():,} params, CPU",
             f"{'shared policy':22s} teacher-forced  note {policy['note_acc']:.3f}"
             f"  next {policy['next_acc']:.3f}"]
    for arm in ARMS:
        s = score(list(zip(eval_exs, run_many(model, eval_exs, arm), strict=True)))
        lines.append(f"{arm:22s} B={DEFAULT_BUDGET[arm]:<3d} acc {_pct(s['answer_acc'])}"
                     f"  walked {_pct(s['walked_ok'])}"
                     f"  misread {_pct(s['wrong_value_rate'])}"
                     f"  lost {_pct(s['lost_rate'])}"
                     f"  over {_pct(s['cite_overclaim'])}"
                     f"  tok {_num(s['reporter_tokens'])}")
    for arm in FRONTIER:
        points = []
        for b in BUDGETS[arm]:
            pairs = list(zip(eval_exs, run_many(model, eval_exs, arm, b), strict=True))
            points.append(f"B={b}:{_pct(score(pairs)['answer_acc'])}")
        curve = "  ".join(points)
        lines.append(f"{arm + ' frontier':22s}       {curve}")
    return "\n".join(lines)


def _pct(value) -> str:
    """`n/a` rather than 0.000 where the arm cannot be asked the question."""
    return "  n/a" if value is None else f"{value:.3f}"


def _num(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _hop_cost(arm: str) -> int:
    """Tokens one hop costs the reporter's window.

    A note arm carries one fact per hop; the arm that skips the notebook carries the
    whole search result, which is the freshest line plus the distractors that came
    back with it.
    """
    return COST[arm] * (1 + DISTRACTORS if arm == "full_context" else 1)


def _arm(text: str) -> str:
    if text not in ARMS:
        raise argparse.ArgumentTypeError(f"expected one of {', '.join(ARMS)}")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="relay")
    sub = parser.add_subparsers(dest="cmd", required=True)

    led = sub.add_parser("ledger", help="fold one query with the exact verifier")
    led.add_argument("--seed", type=int, default=5)
    led.add_argument("--depth", type=int, default=max(EVAL_DEPTHS),
                     choices=EVAL_DEPTHS)
    led.set_defaults(func=_ledger)

    nb = sub.add_parser("notebook", help="show what a token budget carries and loses")
    nb.add_argument("--arm", type=_arm, default="typed_relay")
    nb.add_argument("--budget", type=int, default=2)
    nb.add_argument("--seed", type=int, default=5)
    nb.add_argument("--depth", type=int, default=max(EVAL_DEPTHS),
                    choices=EVAL_DEPTHS)
    nb.set_defaults(func=_notebook)

    demo = sub.add_parser("demo", help="train the shared policy and compare harnesses")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--train", type=int, default=DEMO_TRAIN,
                      help="queries to clone from; the researcher is the part that "
                           "needs them")
    demo.add_argument("--steps", type=int, default=DEMO_STEPS,
                      help="training steps; the defaults are what the README transcript "
                           "was captured at")
    demo.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    if args.cmd == "notebook" and COST[args.arm] and \
            args.budget // _hop_cost(args.arm) > args.depth:
        parser.exit(2, f"--budget {args.budget} is more than {args.depth} hops cost "
                       f"on {args.arm}; the interesting case is a budget that "
                       f"truncates\n")
    print(args.func(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
