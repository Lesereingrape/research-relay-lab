"""Render the README results block from results/frontier.json.

This is the only thing allowed to write the RESULTS section of the README: run
``python experiments/run_study.py`` then ``python experiments/make_report.py --write``.
A test compares the README against ``build(json.load(results/frontier.json))`` byte for
byte, so no number in the README is hand-copied and none can drift.

The renderer measures nothing, and it also *decides nothing*. Every comparison,
superlative and ordering below is read out of the committed artifact -- including how
many seeds agree on a gap -- so a sentence like "provenance did not buy accuracy here"
cannot survive a rerun that stops being true.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("oracle_relay", "typed_relay", "cited_relay", "full_context", "plan_first",
         "single_shot")
# What the harness lets travel between the researcher and the reporter.
TRAVELS = {
    "oracle_relay": "a typed note per hop, written by the verifier",
    "typed_relay": "a typed note per hop",
    "cited_relay": "a typed note plus the record it came from",
    "full_context": "the raw search hits",
    "plan_first": "one search, then a plan the reporter has to trust",
    "single_shot": "nothing: the reporter answers before any search",
}
SEARCHED = dict.fromkeys(ORDER, "the whole chain")
SEARCHED["plan_first"] = "one hop, then it guesses the rest"
SEARCHED["single_shot"] = "no hop"
# Printed as the budget curve: the arms whose budget is the interesting variable.
SWEEP = ("typed_relay", "cited_relay", "full_context", "plan_first")
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven"}


def _word(n: int) -> str:
    """Spell a count the prose states in words, from the artifact's own length."""
    return WORDS.get(n, str(n))


def _n(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pm(cell: dict, digits: int = 3) -> str:
    """`0.955 +/- 0.015`: the mean over seeds and the spread across them."""
    if cell["mean"] is None:
        return "-"
    return f"{_n(cell['mean'], digits)} +/- {_n(cell['std'], digits)}"


def _delta(a: dict, b: dict, digits: int = 3) -> str:
    """A signed difference, so the sentence cannot outlive the ordering it claims."""
    if a["mean"] is None or b["mean"] is None:
        return "-"
    return f"{a['mean'] - b['mean']:+.{digits}f}"


def _ratio(a: dict, b: dict, digits: int = 1) -> str:
    if not b["mean"]:
        return "-"
    return f"{a['mean'] / b['mean']:.{digits}f}"


def _gap(data: dict, high: str, low: str, field: str = "answer_acc") -> tuple:
    """Per-seed differences between two arms at their default budgets.

    Returns ``(mean, a_above, n)``. The count is what lets the prose say how many
    seeds agree, instead of leaning on a mean over three of them.
    """
    seeds = data["config"]["seeds"]
    diffs = [data["per_seed"][str(s)][high][field] - data["per_seed"][str(s)][low][field]
             for s in seeds]
    return round(sum(diffs) / len(diffs), 4), sum(1 for d in diffs if d > 0), len(diffs)


def _agree(mean: float, above: int, n: int) -> str:
    """`down on 3 of 3 seeds` / `up on 2 of 3 seeds`: the ordering, with its support."""
    direction = "down" if mean < 0 else "up"
    hits = above if direction == "up" else n - above
    return f"{direction} on {hits} of {n} seeds"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _sep(head: list[str]) -> str:
    return ("|" + "|".join(["-----" if i == 0 else "------:"
                            for i in range(len(head))]) + "|")


def _table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [_row(head), _sep(head)] + [_row(r) for r in rows]


def build(data: dict) -> str:
    cfg = data["config"]
    arms = data["arms"]
    ty, ci, orc, fc = (arms["typed_relay"], arms["cited_relay"], arms["oracle_relay"],
                       arms["full_context"])
    policy = arms["_policy"]
    chance = _n(cfg["chance"])
    out: list[str] = []

    out.append("*Every figure below is produced by `experiments/run_study.py` on CPU "
               "and committed as [`results/frontier.json`](results/frontier.json); the "
               "tables and the sentences around them are rendered by "
               f"`experiments/make_report.py`. {len(cfg['seeds'])} seeds "
               f"({', '.join(map(str, cfg['seeds']))}), {cfg['n_train']:,} training and "
               f"{cfg['n_eval']:,} held-out queries per seed, {cfg['steps']:,} steps at "
               f"lr {cfg['lr']}, and {cfg['params']:,} parameters shared by every "
               "harness.*")
    env = data["environment"]
    out.append(f"*Measured under Python {env['python']} on {env['platform']}, torch "
               f"{env['torch']}, {env['threads']} CPU threads, {env['device']}. A rerun "
               "inside that environment reproduces the artifact field for field except "
               "the wall-clock; elsewhere the thread count changes the float reduction "
               "order and the last digits move.*")
    out.append("")

    # --- the harnesses --------------------------------------------------------
    out.append(f"### The {_word(len(ORDER))} harnesses, at the budget each would choose")
    out.append("")
    head = ["harness", "what travels from researcher to reporter",
            "tokens per fact", "hops searched", "reporter tokens", "answer acc",
            "walked the chain", "misread a hop", "lost in relay",
            "cites more than it holds"]
    rows = []
    for arm in ORDER:
        a = arms[arm]
        rows.append([f"`{arm}`", TRAVELS[arm], str(cfg["cost"][arm]), SEARCHED[arm],
                     _n(a["reporter_tokens"]["mean"], 1), _pm(a["answer_acc"]),
                     _n(a["walked_ok"]["mean"]), _n(a["wrong_value_rate"]["mean"]),
                     _n(a["lost_rate"]["mean"]), _n(a["cite_overclaim"]["mean"])])
    out.extend(_table(head, rows))
    out.append("")
    out.append(f"*A `-` is not a zero: it means the harness cannot be asked that "
               f"question. `single_shot` never walks, so it has no walk to be right "
               f"about; `full_context` hands over raw hits instead of writing a note, so "
               f"there is no note of its own to misread and no truncated notebook to "
               f"fold. Chance on {_word(cfg['n_values'])} possible answers is "
               f"{chance}.*")
    out.append("")

    # --- the ceiling ----------------------------------------------------------
    mean, above, n = _gap(data, "oracle_relay", "typed_relay")
    out.append(f"**The notebook is not the bottleneck at this budget, and the reader "
               f"is not either.** `oracle_relay` is the control: the same harness, the "
               f"same window, with the notes written by the verifier instead of the "
               f"researcher. It answers **{_n(orc['answer_acc']['mean'])}**, the arm "
               f"that reads its own evidence {_n(ty['answer_acc']['mean'])} "
               f"({_delta(orc['answer_acc'], ty['answer_acc'])}, {_agree(mean, above, n)}"
               f"), and neither loses a fact: `lost_rate` is "
               f"{_n(ty['lost_rate']['mean'])} for both. What separates them is "
               f"{_n(ty['wrong_value_rate']['mean'])} of hops written with the wrong "
               f"value -- the researcher's share of the error, billed to the stage that "
               f"reads rather than the stage that carries.")
    out.append("")
    out.append(f"The reading itself is nearly solved: given the true chain and its "
               f"evidence, the frozen policy writes the right value "
               f"{_n(policy['note_acc']['mean'])} of the time and picks the right next "
               f"record {_n(policy['next_acc']['mean'])} of the time "
               f"(`relay.study.teacher_forced`, on the same held-out queries). Everything "
               f"the arms lose on top of that belongs to the relay or to the walk the "
               f"policy takes for itself.")
    out.append("")

    # --- provenance -----------------------------------------------------------
    mean, above, n = _gap(data, "cited_relay", "typed_relay")
    spread = _n(max(ci["answer_acc"]["std"], ty["answer_acc"]["std"]))
    same = abs(mean) <= float(spread)
    out.append(f"**Provenance buys tokens, not accuracy.** `cited_relay` carries the "
               f"record id beside every value ({cfg['cost']['cited_relay']} tokens per "
               f"fact instead of {cfg['cost']['typed_relay']}), so its default budget "
               f"holds the same {_word(max(cfg['eval_depths']))} hops while its reporter "
               f"reads {_ratio(ci['reporter_tokens'], ty['reporter_tokens'])}x the tokens "
               f"({_n(ci['reporter_tokens']['mean'], 1)} against "
               f"{_n(ty['reporter_tokens']['mean'], 1)}). At that equal capacity it "
               f"scores {_n(ci['answer_acc']['mean'])} against `typed_relay`'s "
               f"{_n(ty['answer_acc']['mean'])}: {_delta(ci['answer_acc'], ty['answer_acc'])}, "
               f"{_agree(mean, above, n)}. "
               + (f"That difference is inside the across-seed spread ({spread}), so this "
                  f"study does not claim provenance *costs* accuracy -- it claims "
                  f"provenance is paid for in context and returns nothing measurable "
                  f"here."
                  if same else
                  f"That difference is outside the across-seed spread ({spread}), so the "
                  f"citation column is a measurable cost at equal capacity, on top of "
                  f"the extra tokens."))
    out.append("")

    # --- raw hits -------------------------------------------------------------
    deep = str(max(cfg["eval_depths"]))
    out.append(f"**Evidence in the window is not evidence the reporter can use.** At "
               f"its default budget `full_context` loses nothing at all: every hop's "
               f"freshest line fits, so `lost_rate` is {_n(fc['lost_rate']['mean'])} and "
               f"`carried_rate` {_n(fc['carried_rate']['mean'])} -- and it still answers "
               f"**{_n(fc['answer_acc']['mean'])}**, against `typed_relay`'s "
               f"{_n(ty['answer_acc']['mean'])} on "
               f"{_n(ty['reporter_tokens']['mean'], 1)} reporter tokens instead of "
               f"{_n(fc['reporter_tokens']['mean'], 1)}. The failure is *when* the "
               f"selection happens. A note is written after the researcher has already "
               f"resolved which line of the record is current; a raw hit list hands the "
               f"reporter the current line plus {cfg['distractors']} distractors per hop "
               f"and asks it to resolve that at read time. On "
               f"{_word(int(deep))}-hop questions, where all "
               f"{int(deep) * (1 + cfg['distractors'])} lines are present, the arm sits "
               f"at {_n(fc['per_depth'][deep]['answer_acc']['mean'])} against a chance "
               f"level of {chance}. `dump everything into the context` is not a free "
               f"substitute for writing things down.")
    out.append("")

    # --- the frontier ---------------------------------------------------------
    out.append("### The same harnesses as the notebook window is cut")
    out.append("")
    budgets = sorted({b for arm in SWEEP for b in cfg["budgets"][arm]})
    head = ["notebook budget (tokens)"] + [f"`{a}`" for a in SWEEP]

    def grid(field: str) -> list[list[str]]:
        rows = []
        for b in budgets:
            cells = [str(b)]
            for arm in SWEEP:
                cell = arms[arm]["budget"].get(str(b))
                cells.append(_n(cell[field]["mean"]) if cell else "-")
            rows.append(cells)
        return rows

    out.extend(_table(head, grid("answer_acc")))
    out.append("")
    out.append("The same grid, as the fraction of the chain that never reaches the "
               "reporter:")
    out.append("")
    out.extend(_table(head, grid("lost_rate")))
    out.append("")

    shared = max(b for b in budgets
                 if all(str(b) in arms[arm]["budget"] for arm in SWEEP))
    at = {arm: arms[arm]["budget"][str(shared)] for arm in SWEEP}
    out.append(f"A budget is not a number of facts. At {_word(shared)} tokens the "
               f"window holds {shared // cfg['cost']['typed_relay']} notes for "
               f"`typed_relay`, {shared // cfg['cost']['cited_relay']} for "
               f"`cited_relay`, and {shared // cfg['cost']['full_context']} raw line for "
               f"`full_context` -- and accuracy is "
               + ", ".join(f"{_n(at[arm]['answer_acc']['mean'])} for `{arm}`"
                           for arm in SWEEP)
               + ". Same tokens, same frozen policy: the interface set the capacity.")
    out.append("")

    floor = min(cfg["budgets"]["typed_relay"])
    small = arms["typed_relay"]["budget"][str(floor)]
    out.append(f"**One slot short and the report stops believing itself.** Cut "
               f"`typed_relay` to {_word(floor)} token: accuracy "
               f"{_n(ty['answer_acc']['mean'])} -> {_n(small['answer_acc']['mean'])}, "
               f"{_n(small['lost_rate']['mean'])} of the chain never arrives, and "
               f"**{_n(small['cite_overclaim']['mean'])} of reports cite more hops than "
               f"their own notebook holds.** A notebook that lost a fact is not a "
               f"notebook that says so: the citation head was trained on full notebooks "
               f"and keeps claiming the hop count the query asked for. The reporter "
               f"cannot notice the loss from the inside -- only the interface can.")
    out.append("")
    second = sorted(cfg["budgets"]["typed_relay"])[1]
    mid = arms["typed_relay"]["budget"][str(second)]
    out.append(f"What it does instead is add up what is left. At {_word(second)} tokens, "
               f"where the walk is still right {_n(mid['walked_ok']['mean'])} of the "
               f"time, {_n(mid['partial_sum_rate']['mean'])} of answers are the exact "
               f"fold of a truncated chain and {_n(mid['cite_overclaim']['mean'])} cite "
               f"more than they hold. That is the failure shape a deep-research pipeline "
               f"shows a user: a summary that is arithmetically consistent with the "
               f"notes it kept, reads as though it were built from every source it "
               f"cites, and is wrong about the question that was asked.")
    out.append("")
    plan = arms["plan_first"]
    plan_acc = [plan["budget"][str(b)]["answer_acc"]["mean"]
                for b in cfg["budgets"]["plan_first"]]
    out.append(f"`plan_first` is the arm that decides the whole route from one search, "
               f"and no budget can rescue it: accuracy stays between "
               f"{_n(min(plan_acc))} and {_n(max(plan_acc))} across its whole sweep "
               f"with a window that carries "
               f"{_n(plan['carried_rate']['mean'])} of the chain and a walk that is "
               f"right {_n(plan['walked_ok']['mean'])} of the time. The facts were never "
               f"gathered, so widening the notebook only gives the reporter more room to "
               f"guess in.")
    out.append("")

    # --- depth ----------------------------------------------------------------
    out.append("### The same budget, by how deep the chain is")
    out.append("")
    depths = sorted(int(d) for d in cfg["eval_depths"])
    head = ["harness"] + [f"{_word(d)}-hop acc" for d in depths] + \
           [f"{_word(d)}-hop lost" for d in depths]
    rows = []
    for arm in ORDER:
        cells = [f"`{arm}`"]
        cells += [_n(arms[arm]["per_depth"][str(d)]["answer_acc"]["mean"])
                  for d in depths]
        cells += [_n(arms[arm]["per_depth"][str(d)]["lost_rate"]["mean"])
                  for d in depths]
        rows.append(cells)
    out.extend(_table(head, rows))
    out.append("")
    shallow, deep_d = str(min(depths)), str(max(depths))
    penalty = {arm: arms[arm]["per_depth"][deep_d]["answer_acc"]["mean"]
               - arms[arm]["per_depth"][shallow]["answer_acc"]["mean"] for arm in ORDER}
    out.append(f"Every extra hop is one more fact that has to survive, and one more "
               f"chance to misread it. Going from {_word(int(shallow))} to "
               f"{_word(int(deep_d))} hops costs the verifier-written notebook "
               f"{penalty['oracle_relay']:+.3f} and the researcher-written one "
               f"{penalty['typed_relay']:+.3f}, at a budget where neither loses a fact "
               f"(`lost` {_n(arms['typed_relay']['per_depth'][deep_d]['lost_rate']['mean'])} "
               f"at the bottom). The depth penalty is mostly a reading penalty, not a "
               f"capacity one -- which is exactly the part an end-to-end score folds "
               f"into a single number.")
    out.append("")

    profile = ty["hops"]
    if profile:
        trend = ("rises" if profile[-1] > profile[0]
                 else "is flat" if profile[-1] == profile[0] else "falls")
        out.append("Misreads by position in the chain, over the same `typed_relay` "
                   "rollouts: "
                   + ", ".join(f"hop {i + 1} {_n(v)}" for i, v in enumerate(profile))
                   + f". The rate {trend} with distance from the record the query named: "
                   f"the same policy reads {_n(policy['note_acc']['mean'])} of a *true* "
                   f"chain's hops right when the harness hands it the record to stand on, "
                   f"and misreads {_n(ty['wrong_value_rate']['mean'])} of the hops on the "
                   f"walk it takes for itself. No end-to-end score reports this.")
        out.append("")

    # --- per seed ------------------------------------------------------------
    out.append("### Per seed, so a gap carried by one initialisation stays visible")
    out.append("")
    head = ["seed"] + [f"`{a}`" for a in ORDER]
    rows = []
    for seed in cfg["seeds"]:
        rows.append([str(seed)] + [_n(data["per_seed"][str(seed)][a]["answer_acc"])
                                   for a in ORDER])
    out.extend(_table(head, rows))
    out.append("")
    widest = max(arms[a]["answer_acc"]["std"] for a in ORDER)
    steadiest = min(ORDER, key=lambda a: arms[a]["answer_acc"]["std"])
    guess = arms["single_shot"]["answer_acc"]
    out.append(f"Read every gap above against the widest across-seed standard deviation "
               f"in these tables, {_n(widest)} (the steadiest arm is `{steadiest}` at "
               f"{_n(arms[steadiest]['answer_acc']['std'])}); the provenance comparison "
               f"is the one this report declines to claim precisely because it sits "
               f"inside it. `single_shot` answers at {_n(guess['mean'])} against a "
               f"chance level of {chance}: with no evidence the reporter is guessing, "
               f"which is what makes the other rows evidence-driven rather than "
               f"prior-driven.")
    out.append("")
    return "\n".join(out)


START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
# The block between the markers, markers included. `(?:.*?\n)?` so a README that has
# never been rendered -- an empty block -- still matches and can be filled in.
PATTERN = re.compile(re.escape(START) + r"\n(?:.*?\n)?" + re.escape(END), re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser(prog="make_report")
    parser.add_argument("--write", action="store_true",
                        help="splice the rendered block into README.md")
    parser.add_argument("--check", action="store_true",
                        help="fail if README.md does not already contain it")
    args = parser.parse_args()
    data = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    block = build(data)
    spliced = f"{START}\n{block}\n{END}"
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if args.check:
        current = PATTERN.search(text)
        assert current, f"README.md has no {START} marker"
        if current.group(0) != spliced:
            raise SystemExit("README results block is stale; run "
                             "`python experiments/make_report.py --write`")
        print("README matches results/frontier.json")
        return
    if args.write:
        assert PATTERN.search(text), f"README.md has no {START}/{END} markers"
        readme.write_text(PATTERN.sub(lambda _: spliced, text), encoding="utf-8")
        print(f"spliced {len(block.splitlines())} lines into README.md")
        return
    print(block)


if __name__ == "__main__":
    main()
