"""The artifact has to agree with itself, and with the code that produced it.

A README block can only be trusted as far as the JSON behind it. These tests re-derive
what they can from what the JSON itself stores -- the headline from the per-seed rows,
the default-budget column from the budget sweep, the policy means from the per-seed
policy -- and then check the arithmetic the harness owes regardless of what any model
did: a notebook either carries a hop or loses it, a bigger window never loses more, and
a chain nothing truncated cannot come out as a partial sum.

Nothing here reruns the study. Every invariant is a relationship between numbers that
are already in the file, so a hand-edited artifact trips on it.
"""

from __future__ import annotations

import json
import statistics
from itertools import pairwise
from pathlib import Path

from relay.harness import (
    ARMS,
    BUDGETS,
    COST,
    DEFAULT_BUDGET,
    DISTRACTORS,
    HEAD_PREFIX,
    MAX_PAYLOAD,
    NOTE_ARMS,
)
from relay.kb import MAX_DEPTH, MIN_DEPTH, N_REC, N_TOK, N_VAL, VERSIONS
from relay.metrics import FIELDS
from relay.model import count_parameters
from relay.study import EVAL_DEPTHS, HOP_ARMS, N_EVAL, N_TRAIN, SEEDS
from relay.train import BATCH, LR, STEPS

ROOT = Path(__file__).resolve().parents[1]
ART = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
CONFIG = ART["config"]
AR = ART["arms"]
PER_SEED = ART["per_seed"]


def _stats(values: list) -> dict:
    """The same rounding rule `study._stats` applies, written out so it can disagree."""
    seen = [v for v in values if v is not None]
    if not seen:
        return {"mean": None, "std": None}
    return {"mean": round(statistics.fmean(seen), 4),
            "std": round(statistics.pstdev(seen), 4)}


def _rows(arm: str) -> list[dict]:
    """Every aggregated row the artifact holds for one arm."""
    return [AR[arm], *AR[arm]["budget"].values(), *AR[arm]["per_depth"].values()]


def test_the_artifact_ships_every_arm_and_every_field_the_code_defines():
    assert CONFIG["arms"] == list(ARMS)
    assert set(AR) == {*ARMS, "_policy"}
    for arm in ARMS:
        assert set(AR[arm]) == {*FIELDS, "per_depth", "budget", "hops"}
        assert [int(k) for k in AR[arm]["per_depth"]] == list(EVAL_DEPTHS)
        assert sorted(int(k) for k in AR[arm]["budget"]) == sorted(
            CONFIG["budgets"][arm])
        for row in [*AR[arm]["budget"].values(), *AR[arm]["per_depth"].values()]:
            assert set(row) == set(FIELDS)


def test_the_per_seed_rows_are_the_arms_and_the_headline_fields():
    assert set(PER_SEED) == {str(s) for s in SEEDS}
    for seed, arms in PER_SEED.items():
        assert set(arms) == set(ARMS), f"seed {seed}"
        for arm, row in arms.items():
            # `acc_when_walked` is a conditional mean over a subset, so it is not
            # averageable across seeds and `run_study` does not ship it per seed.
            assert set(row) == set(FIELDS) - {"acc_when_walked"}, f"{seed}/{arm}"


def test_the_headline_is_the_mean_and_spread_of_the_seeds_it_prints_beside():
    fields = sorted(PER_SEED[str(SEEDS[0])][ARMS[0]])
    for arm in ARMS:
        for field in fields:
            assert AR[arm][field] == _stats([PER_SEED[str(s)][arm][field]
                                             for s in SEEDS]), (
                f"{arm}/{field} is not the mean and spread of the per-seed rows")


def test_the_policy_block_is_the_per_seed_policy_averaged():
    assert set(ART["per_seed_policy"]) == {str(s) for s in SEEDS}
    for name, block in AR["_policy"].items():
        assert block == _stats([p[name]
                                for p in ART["per_seed_policy"].values()])


def test_the_default_budget_column_is_a_row_of_the_sweep_that_ran():
    """One run at the default budget, so the headline and its own sweep cannot diverge."""
    for arm in ARMS:
        default = DEFAULT_BUDGET[arm]
        assert default in CONFIG["budgets"][arm]
        assert str(default) in AR[arm]["budget"]
        for field in FIELDS:
            assert AR[arm][field] == AR[arm]["budget"][str(default)][field]


def test_the_per_depth_rows_bracket_the_headline():
    """A mean over disjoint query groups has to sit between the group means."""
    for arm in ARMS:
        for field in ("answer_acc", "walked_ok", "lost_rate"):
            rows = [d[field]["mean"] for d in AR[arm]["per_depth"].values()]
            rows = [r for r in rows if r is not None]
            mean = AR[arm][field]["mean"]
            if rows and mean is not None:
                assert min(rows) - 1e-4 <= mean <= max(rows) + 1e-4, (
                    f"{arm}/{field}: the depth split is not a partition of the headline")


def test_a_hop_is_either_carried_or_lost_and_neither_more():
    for arm in ARMS:
        for row in _rows(arm):
            carried, lost = row["carried_rate"]["mean"], row["lost_rate"]["mean"]
            if carried is None or lost is None:
                continue
            assert abs(carried + lost - 1.0) <= 1e-4, (
                f"{arm}: carried {carried} and lost {lost} do not make one chain")


def test_a_wider_notebook_never_loses_more():
    for arm in ARMS:
        if arm == "single_shot":
            continue
        budgets = [int(b) for b in AR[arm]["budget"]]
        lost = [AR[arm]["budget"][str(b)]["lost_rate"]["mean"]
                for b in sorted(budgets)]
        assert all(a >= b - 1e-9 for a, b in pairwise(lost)), (
            f"{arm}: a larger budget lost more of the chain: "
            f"{dict(zip(budgets, lost, strict=True))}")


def test_the_reporter_never_reads_fewer_tokens_at_a_bigger_budget():
    for arm in ARMS:
        budgets = [int(b) for b in AR[arm]["budget"]]
        tokens = [AR[arm]["budget"][str(b)]["reporter_tokens"]["mean"]
                  for b in sorted(budgets)]
        assert all(a <= b + 1e-9 for a, b in pairwise(tokens)), (
            f"{arm}: {dict(zip(budgets, tokens, strict=True))}")
        assert all(HEAD_PREFIX <= t <= HEAD_PREFIX + MAX_PAYLOAD for t in tokens), (
            f"{arm}: the report is the head plus a payload that fits: {tokens}")


def test_a_chain_that_arrived_whole_is_not_a_partial_sum():
    """Truncation is the only thing that can make the fold come out short."""
    for arm in NOTE_ARMS:
        for row in AR[arm]["budget"].values():
            if row["lost_rate"]["mean"] == 0.0:
                assert row["partial_sum_rate"]["mean"] == 0.0
                assert row["acc_full_notebook"]["mean"] == row["answer_acc"]["mean"]


def test_only_an_arm_with_a_notebook_is_scored_on_what_its_notebook_held():
    for arm in ARMS:
        row = AR[arm]
        notebook_arm = arm in NOTE_ARMS
        for field in ("partial_sum_rate", "wrong_value_rate"):
            assert (row[field]["mean"] is None) == (not notebook_arm), f"{arm}/{field}"
        # The ceiling is only a number for an arm that ever kept a whole chain, and
        # `plan_first` never does at any budget -- so `None`, not 0.000, is the honest
        # thing to print beside an accuracy that came from a truncated notebook.
        kept = notebook_arm and any(r["lost_rate"]["mean"] == 0.0
                                    for r in AR[arm]["budget"].values())
        assert (row["acc_full_notebook"]["mean"] is not None) == kept, arm
        assert (row["walked_ok"]["mean"] is None) == (arm == "single_shot"), arm
        assert (row["hops"] is not None) == (arm in HOP_ARMS), arm


def test_single_shot_reports_a_fact_it_cannot_have():
    """No search, so nothing arrived, so any citation is over-claimed by construction."""
    row = AR["single_shot"]
    assert row["lost_rate"]["mean"] == 1.0
    assert row["carried_rate"]["mean"] == 0.0
    assert row["cite_overclaim"]["mean"] == 1.0


def test_the_sweep_starts_where_a_hop_does_not_fit():
    """The point of the sweep is the cliff, so the smallest budget has to truncate."""
    for arm in ("typed_relay", "cited_relay", "oracle_relay"):
        smallest = min(CONFIG["budgets"][arm])
        assert smallest // COST[arm] < MIN_DEPTH, (
            f"{arm}: even budget {smallest} holds every chain, so the sweep cannot show "
            "a loss")
        assert AR[arm]["budget"][str(smallest)]["lost_rate"]["mean"] > 0.0


def test_the_config_records_the_constants_the_code_actually_uses():
    assert CONFIG["seeds"] == list(SEEDS)
    assert CONFIG["n_train"] == N_TRAIN and CONFIG["n_eval"] == N_EVAL
    assert CONFIG["eval_depths"] == list(EVAL_DEPTHS)
    assert [CONFIG["min_depth"], CONFIG["max_depth"]] == [MIN_DEPTH, MAX_DEPTH]
    assert CONFIG["steps"] == STEPS and CONFIG["batch"] == BATCH
    assert CONFIG["lr"] == LR
    assert CONFIG["params"] == count_parameters()
    assert CONFIG["distractors"] == DISTRACTORS
    assert [CONFIG["n_tokens"], CONFIG["n_records"], CONFIG["n_values"]] == [
        N_TOK, N_REC, N_VAL]
    assert CONFIG["versions"] == list(VERSIONS)
    assert CONFIG["max_payload"] == MAX_PAYLOAD
    assert CONFIG["cost"] == COST
    assert CONFIG["default_budget"] == DEFAULT_BUDGET
    assert {a: list(BUDGETS[a]) for a in ARMS} == CONFIG["budgets"]


def test_chance_is_the_random_guess_the_tables_compare_against():
    assert CONFIG["chance"] == round(1.0 / N_VAL, 4)
    assert AR["single_shot"]["answer_acc"]["mean"] <= CONFIG["chance"] + 0.01


def test_the_environment_block_is_the_one_a_rerun_has_to_match():
    assert set(ART["environment"]) == {"python", "platform", "torch", "threads",
                                       "device"}
    assert ART["environment"]["device"] == "cpu"
    assert int(ART["environment"]["threads"]) >= 1
    assert float(ART["runtime_sec"]) > 0
