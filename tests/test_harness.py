"""The harnesses, checked as bookkeeping rather than as results.

Each arm is a rule about what travels from researcher to reporter. A rule that is
subtly wrong -- a notebook that drops the wrong end of the chain, a budget that
counts tokens the reporter never sees -- would move every number in the README while
still printing a plausible table. So these tests assert the bookkeeping against hand
computations on an untrained policy, and pin the one currency the arms share: a typed
note is one token, the same note with its provenance is two, a raw hit is four.
"""

from __future__ import annotations

import pytest

from relay.harness import (
    ARMS,
    BUDGETS,
    COST,
    DEFAULT_BUDGET,
    DISTRACTORS,
    HEAD_PREFIX,
    MAX_PAYLOAD,
    QUERY_ROLE,
    Trace,
    _visible_fresh,
    batch,
    chain_records,
    encode,
    lines_payload,
    look,
    notebook,
    run,
    run_many,
)
from relay.kb import (
    N_VAL,
    NONE,
    RECS,
    SEP,
    VALS,
    make_dataset,
    tok,
)
from relay.model import MAXLEN, build
from relay.study import EVAL_DEPTHS

EXS = make_dataset(24, 0, EVAL_DEPTHS)
MODEL = build(0)
# The notebook arms that walk the whole chain; `plan_first` deliberately searches once.
SEQUENTIAL = ("typed_relay", "cited_relay", "oracle_relay")


def test_every_arm_has_a_cost_and_a_budget_grid():
    assert set(COST) == set(DEFAULT_BUDGET) == set(BUDGETS) == set(ARMS)
    for arm in ARMS:
        assert BUDGETS[arm]
        if arm != "single_shot":
            assert DEFAULT_BUDGET[arm] in BUDGETS[arm]


def test_the_token_cost_is_the_three_interfaces_stated_difference():
    """One typed value, the same value with the record it came from, or the raw hit."""
    assert COST["typed_relay"] == 1
    assert COST["cited_relay"] == 2
    assert COST["full_context"] == 4
    assert COST["single_shot"] == 0


@pytest.mark.parametrize("arm", ARMS)
def test_the_batched_rollout_agrees_with_the_one_query_rollout(arm):
    """`run_many` exists to make the study finish; it may not change the answers."""
    single = [run(MODEL, ex, arm) for ex in EXS]
    many = run_many(MODEL, EXS, arm)
    assert len(single) == len(many) == len(EXS)
    for a, b in zip(single, many, strict=True):
        assert (a.answer, a.cite, a.notes, a.visited, a.oks, a.tokens) == \
               (b.answer, b.cite, b.notes, b.visited, b.oks, b.tokens)


@pytest.mark.parametrize("arm", ARMS)
def test_a_row_never_exceeds_the_positional_embedding(arm):
    for ex in EXS:
        for budget in BUDGETS[arm]:
            assert run(MODEL, ex, arm, budget).tokens <= MAXLEN
    assert len(encode(QUERY_ROLE, EXS[0].start, EXS[0], [0] * (MAX_PAYLOAD + 9))) \
        == MAXLEN


@pytest.mark.parametrize("arm", SEQUENTIAL)
def test_a_budget_the_chain_fits_carries_every_hop(arm):
    for ex in EXS:
        tr = run(MODEL, ex, arm, ex.depth * COST[arm])
        assert tr.carried == ex.depth
        assert tr.lost == 0
        assert None not in tr.notes


@pytest.mark.parametrize("arm", SEQUENTIAL)
def test_a_budget_one_fact_short_drops_the_tail(arm):
    """The notebook is written in hop order and clipped at the end, so what goes
    missing is the far end of the chain -- the part a reader cannot re-derive."""
    for ex in EXS:
        capacity = ex.depth - 1
        tr = run(MODEL, ex, arm, capacity * COST[arm])
        assert tr.carried == capacity and tr.lost == 1
        assert len(tr.notes) == capacity
        assert tr.tokens == HEAD_PREFIX + capacity * COST[arm]
        if arm == "oracle_relay":
            assert tr.notes == ex.chain[:capacity]


def test_oracle_notes_are_the_true_chain_by_construction():
    """The control arm's whole job is to hand the reporter a flawless notebook."""
    for ex in EXS:
        tr = run(MODEL, ex, "oracle_relay", ex.depth * COST["oracle_relay"])
        assert tr.notes[:ex.depth] == ex.chain


def test_a_cited_notebook_pays_two_tokens_per_fact_and_names_the_record():
    notes, visited = [1, 3], [7, 9]
    cited = notebook(notes, visited, COST["cited_relay"])
    assert cited == [tok(VALS[1]), tok(RECS[7]), tok(VALS[3]), tok(RECS[9])]
    assert len(cited) == 2 * COST["cited_relay"]
    assert len(notebook(notes, visited, COST["typed_relay"])) == 2


def test_a_blank_slot_is_a_blank_token_and_not_a_zero_value():
    payload = notebook([2, None], [3, 4], COST["typed_relay"])
    assert payload == [tok(VALS[2]), tok(NONE)]
    assert payload[1] != tok(VALS[0])


def test_a_search_returns_the_freshest_line_buried_among_distractors():
    ex = EXS[0]
    hits = look(ex, ex.start, 0)
    assert len(hits) == 1 + DISTRACTORS
    assert ex.corpus.fresh(ex.start) in hits
    assert {h.rec for h in hits} >= {ex.start}


def test_the_hit_order_varies_with_the_salt_while_the_content_does_not():
    ex = EXS[0]
    assert len({tuple(h.rec for h in look(ex, ex.start, s)) for s in range(8)}) > 1
    assert all(ex.corpus.fresh(ex.start) in look(ex, ex.start, s) for s in range(8))
    assert look(ex, ex.start, 2) == look(ex, ex.start, 2)


def test_single_shot_pays_the_head_and_nothing_else():
    for ex in EXS:
        tr = run(MODEL, ex, "single_shot")
        assert tr.tokens == HEAD_PREFIX
        assert tr.carried == 0 and tr.lost == ex.depth
        assert tr.oks == () and tr.notes == ()


def test_plan_first_owes_the_whole_chain_on_one_search():
    for ex in EXS:
        tr = run(MODEL, ex, "plan_first", ex.depth * COST["plan_first"])
        assert len(tr.oks) == 1
        assert tr.wanted == ex.depth and tr.carried == 1
        assert tr.visited == (ex.start,) * 1 + (tr.visited[1],) * (ex.depth - 1)


def test_full_context_carries_a_hop_only_while_its_fresh_line_survives_the_clip():
    for ex in EXS:
        whole = run(MODEL, ex, "full_context", DEFAULT_BUDGET["full_context"])
        assert whole.carried == ex.depth
        one_line = run(MODEL, ex, "full_context", COST["full_context"])
        assert one_line.carried <= 1
        assert one_line.tokens == HEAD_PREFIX + COST["full_context"]


def test_visible_fresh_counts_hops_not_lines():
    """A search returns its hits in a shuffled order, so a clip can hold a whole line
    about some unrelated record and still carry no hop at all."""
    for ex in EXS:
        recs = chain_records(ex)
        every_fresh = lines_payload([ex.corpus.fresh(r) for r in recs])
        assert _visible_fresh(every_fresh, ex, recs) == ex.depth
        wrong_hop = lines_payload([ex.corpus.fresh(recs[1])])
        assert _visible_fresh(wrong_hop, ex, [recs[0]]) == 0


def test_a_partial_sum_is_the_fold_of_surviving_notes_and_nothing_else():
    ex = EXS[0]
    truncated = Trace(arm="typed_relay", budget=ex.depth - 1, depth=ex.depth,
                      answer=sum(ex.chain[:-1]) % N_VAL, cite=ex.depth, tokens=7,
                      notes=tuple(ex.chain[:-1]), wanted=ex.depth,
                      carried=ex.depth - 1)
    assert truncated.partial_sum and truncated.lost == 1
    complete = Trace(arm="typed_relay", budget=ex.depth, depth=ex.depth,
                     answer=ex.answer, cite=ex.depth, tokens=7,
                     notes=ex.chain, wanted=ex.depth, carried=ex.depth)
    assert not complete.partial_sum


def test_carried_plus_lost_is_always_the_chain_owed():
    for arm in ARMS:
        for ex in EXS:
            for budget in BUDGETS[arm]:
                tr = run(MODEL, ex, arm, budget)
                assert tr.carried + tr.lost == max(tr.wanted, tr.carried)
                assert tr.carried <= ex.depth and tr.lost <= ex.depth


def test_chain_records_recomputes_the_walk_without_the_model():
    for ex in EXS:
        recs = chain_records(ex)
        assert recs[0] == ex.start and len(recs) == ex.depth
        assert recs == list(ex.records)


def test_a_line_is_four_tokens_and_ends_with_the_separator():
    ex = EXS[0]
    payload = lines_payload(look(ex, ex.start, 0))
    assert len(payload) == 4 * (1 + DISTRACTORS)
    assert payload[-1] == tok(SEP)


def test_batch_pads_to_the_widest_row_and_returns_one_logit_row_each():
    out = batch(MODEL, [[1, 2, 3], [4, 5]])
    assert out.note.shape == (2, N_VAL)
    assert out.nxt.shape[1] == len(RECS)
