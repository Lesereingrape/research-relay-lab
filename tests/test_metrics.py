"""The decomposition, checked on traces a test writes by hand.

`answer_acc` alone would let a wrong answer look like a wrong researcher. The point of
these fields is that they answer to different stages, so the tests build traces whose
stage of failure is fixed by construction and ask whether the scorer blames the right
one -- including whether it stays silent when an arm cannot be asked the question.
"""

from __future__ import annotations

import pytest

from relay.harness import COST, HEAD_PREFIX, Trace
from relay.kb import N_VAL, make_dataset
from relay.metrics import FIELDS, by_field, hop_profile, score
from relay.study import EVAL_DEPTHS

EX = make_dataset(4, 2, EVAL_DEPTHS)


def _pair(arm: str, ex, *, answer: int, cite: int, carried: int | None = None,
          wanted: int | None = None, walked: bool = False, wrong: int = 0,
          notes: tuple | None = None, oks: tuple = (), tokens: int = HEAD_PREFIX):
    tr = Trace(arm=arm, budget=3, depth=ex.depth, answer=answer, cite=cite,
               tokens=tokens, visited=ex.records, notes=notes or (), oks=oks,
               wanted=ex.depth if wanted is None else wanted,
               carried=ex.depth if carried is None else carried,
               walked_ok=walked, wrong_value=wrong)
    return ex, tr


def test_an_empty_pile_has_no_numbers_rather_than_zeroes():
    assert score([]) == dict.fromkeys(FIELDS)


def test_a_complete_notebook_and_an_honest_citation_lose_nothing():
    ex = EX[0]
    out = score([_pair("typed_relay", ex, answer=ex.answer, cite=ex.depth,
                       notes=ex.chain)])
    assert out["lost_rate"] == 0.0
    assert out["carried_rate"] == 1.0
    assert out["cite_overclaim"] == 0.0
    assert out["answer_acc"] == 1.0
    assert out["acc_full_notebook"] == 1.0
    assert out["partial_sum_rate"] == 0.0


def test_a_citation_that_matches_the_notebook_is_not_an_overclaim():
    ex = EX[0]
    carried = ex.depth - 1
    out = score([_pair("typed_relay", ex, answer=0, cite=carried, carried=carried,
                       notes=ex.chain[:carried])])
    assert out["cite_overclaim"] == 0.0
    assert out["lost_rate"] == pytest.approx(carried_only(carried, ex.depth))


def carried_only(carried: int, depth: int) -> float:
    return (depth - carried) / depth


def test_a_report_that_cites_more_hops_than_it_carries_is_caught():
    ex = EX[0]
    out = score([_pair("typed_relay", ex, answer=ex.answer, cite=ex.depth, carried=1,
                       notes=ex.chain[:1])])
    assert out["cite_overclaim"] == 1.0
    assert out["carried_rate"] == pytest.approx(1 / ex.depth)


def test_folding_what_survived_is_scored_as_a_partial_sum():
    ex = EX[0]
    kept = ex.chain[:-1]
    out = score([_pair("cited_relay", ex, answer=sum(kept) % N_VAL, cite=ex.depth,
                       carried=len(kept), notes=kept)])
    assert out["partial_sum_rate"] == 1.0
    assert out["answer_acc"] == float(sum(kept) % N_VAL == ex.answer)


def test_a_wrong_answer_from_a_wrong_read_is_blamed_on_the_reader():
    ex = EX[0]
    out = score([_pair("typed_relay", ex, answer=ex.answer + 1, cite=ex.depth,
                       walked=True, wrong=1, notes=ex.chain)])
    assert out["lost_rate"] == 0.0
    assert out["wrong_value_rate"] == pytest.approx(1 / ex.depth)
    assert out["walked_ok"] == 1.0
    assert out["acc_when_walked"] == 0.0


def test_an_arm_that_never_walked_is_not_scored_as_a_bad_walk():
    ex = EX[0]
    out = score([_pair("single_shot", ex, answer=0, cite=ex.depth, carried=0)])
    assert out["walked_ok"] is None
    assert out["wrong_value_rate"] is None
    assert out["partial_sum_rate"] is None
    assert out["acc_full_notebook"] is None
    assert out["acc_when_walked"] is None
    assert out["lost_rate"] == 1.0
    assert out["reporter_tokens"] == float(HEAD_PREFIX)


def test_an_arm_without_a_notebook_is_not_scored_on_one():
    ex = EX[0]
    out = score([_pair("full_context", ex, answer=ex.answer, cite=ex.depth,
                       carried=ex.depth, walked=True,
                       tokens=HEAD_PREFIX + COST["full_context"])])
    assert out["acc_full_notebook"] is None
    assert out["wrong_value_rate"] is None
    assert out["partial_sum_rate"] is None
    assert out["walked_ok"] == 1.0
    assert out["reporter_tokens"] == HEAD_PREFIX + COST["full_context"]


def test_a_mean_is_taken_over_queries_not_over_arms():
    pairs = [_pair("typed_relay", ex, answer=ex.answer, cite=ex.depth, notes=ex.chain)
             for ex in EX[:2]]
    pairs.append(_pair("typed_relay", EX[2], answer=EX[2].answer + 1, cite=EX[2].depth,
                       notes=EX[2].chain))
    assert score(pairs)["answer_acc"] == round(2 / 3, 4)
    assert by_field("answer_acc", pairs) == round(2 / 3, 4)


def test_the_field_order_is_the_table_order():
    assert tuple(score([_pair("typed_relay", EX[0], answer=0, cite=1)])) == FIELDS


def test_a_hop_nobody_reached_has_no_rate():
    ex = EX[0]
    out = hop_profile([_pair("typed_relay", ex, answer=0, cite=0, oks=(True,)),
                       _pair("typed_relay", ex, answer=0, cite=0,
                             oks=(True, False, False))])
    assert out == [0.0, 1.0, 1.0]


def test_a_hop_rate_is_a_mean_over_the_queries_that_reached_it():
    ex = EX[0]
    out = hop_profile([_pair("typed_relay", ex, answer=0, cite=0, oks=(False,)),
                       _pair("typed_relay", ex, answer=0, cite=0, oks=(True,))])
    assert out == [0.5]
