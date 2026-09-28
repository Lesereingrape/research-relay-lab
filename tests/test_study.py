"""The study loop, run once at a size a test can afford.

These are structural checks on a real -- but tiny -- training run: that every arm is
scored, that every budget in the sweep grid is measured, that the per-seed numbers and
the aggregate agree, and that the invariant `carried + lost == what the report owed`
survives the aggregation rather than only holding inside one trace. A silently missing
budget cell or an arm that stopped being evaluated would otherwise reach the README as
an empty table row.
"""

from __future__ import annotations

import pytest

from relay.harness import ARMS, BUDGETS, HEAD_PREFIX
from relay.kb import MIN_DEPTH
from relay.metrics import FIELDS
from relay.study import (
    EVAL_DEPTHS,
    HOP_ARMS,
    N_EVAL,
    N_TRAIN,
    SEEDS,
    _stats,
    aggregate,
    run_seed,
)

MAX_DEPTH = max(EVAL_DEPTHS)


def test_the_study_replicates_over_seeds_and_evaluates_two_depths():
    """The README's claims about spread rest on these being real counts."""
    assert len(SEEDS) >= 3
    assert N_EVAL >= 1000 and N_TRAIN >= 2000
    assert EVAL_DEPTHS == (MIN_DEPTH, MIN_DEPTH + 1)


def test_stats_ignores_the_seeds_that_could_not_answer():
    assert _stats([0.5, 0.7]) == {"mean": 0.6, "std": 0.1}
    assert _stats([0.5, None, 0.7]) == {"mean": 0.6, "std": 0.1}
    assert _stats([None, None]) == {"mean": None, "std": None}
    assert _stats([1 / 3])["mean"] == pytest.approx(0.3333)


@pytest.fixture(scope="module")
def one_seed() -> dict:
    import relay.study as S

    train_n, eval_n, steps = S.N_TRAIN, S.N_EVAL, S.STEPS
    S.N_TRAIN, S.N_EVAL, S.STEPS = 48, 32, 40
    try:
        return run_seed(0)
    finally:
        S.N_TRAIN, S.N_EVAL, S.STEPS = train_n, eval_n, steps


def test_every_arm_and_every_budget_cell_is_measured(one_seed):
    assert set(one_seed["arms"]) == set(ARMS)
    for arm in ARMS:
        assert set(one_seed["per_depth"][arm]) == {str(d) for d in EVAL_DEPTHS}
        assert set(one_seed["budget"][arm]) == {str(b) for b in BUDGETS[arm]}
        assert set(one_seed["arms"][arm]) == set(FIELDS)


def test_a_seed_records_what_the_policy_itself_could_do(one_seed):
    policy = one_seed["policy"]
    assert set(policy) == {"note_acc", "next_acc"}
    assert all(0.0 <= v <= 1.0 for v in policy.values())


def test_the_hops_profile_exists_only_where_a_notebook_was_written(one_seed):
    assert set(one_seed["hops"]) == set(HOP_ARMS)
    for arm, profile in one_seed["hops"].items():
        assert len(profile) <= MAX_DEPTH
        assert all(v is None or 0.0 <= v <= 1.0 for v in profile), arm


def test_the_aggregate_is_the_mean_of_its_own_seeds(one_seed):
    agg = aggregate([one_seed])
    for arm in ARMS:
        for name in FIELDS:
            value = one_seed["arms"][arm][name]
            if value is None:
                assert agg[arm][name]["mean"] is None, f"{arm} {name}"
            else:
                assert agg[arm][name]["mean"] == pytest.approx(value), f"{arm} {name}"
        assert agg[arm]["per_depth"] and agg[arm]["budget"]
        assert (agg[arm]["hops"] is None) == (arm not in HOP_ARMS)
    assert agg["_policy"]["note_acc"]["std"] == 0.0


def test_carried_and_lost_partition_what_the_report_owed(one_seed):
    for arm in ARMS:
        for name, cell in one_seed["arms"][arm].items():
            if name in ("carried_rate", "lost_rate"):
                assert cell is None or 0.0 <= cell <= 1.0, f"{arm} {name}"
        carried, lost = one_seed["arms"][arm]["carried_rate"], \
            one_seed["arms"][arm]["lost_rate"]
        assert carried + lost == pytest.approx(1.0, abs=2e-4), arm
        assert one_seed["arms"][arm]["reporter_tokens"] >= HEAD_PREFIX


def test_an_unmeasurable_condition_stays_unmeasurable_through_aggregation(one_seed):
    assert one_seed["arms"]["single_shot"]["walked_ok"] is None
    assert aggregate([one_seed])["single_shot"]["walked_ok"]["mean"] is None
    assert one_seed["arms"]["full_context"]["wrong_value_rate"] is None
    assert one_seed["arms"]["single_shot"]["partial_sum_rate"] is None
