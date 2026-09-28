"""The task has an exact answer, and nothing about that answer involves a model.

If `truth()` were wrong, every number in the README would be measuring agreement with
a mistake. These tests recompute the fold by hand, check the properties the harness
relies on (a fresh line never self-loops, stale versions really are stale), and pin
the one design decision that makes the study mean anything: two queries never share a
ledger, so a policy cannot answer by remembering record 7.
"""

from __future__ import annotations

import random

import pytest

from relay.kb import (
    DEPTHS,
    I2T,
    MIN_DEPTH,
    N_CITE,
    N_REC,
    N_TOK,
    N_VAL,
    RECS,
    TOK,
    VALS,
    dec,
    make_corpus,
    make_dataset,
    make_example,
    tok,
    truth,
)
from relay.study import EVAL_DEPTHS


def test_the_answer_is_the_fold_of_the_values_the_chain_walks():
    ex = make_example(random.Random(5), 3)
    assert ex.answer == sum(ex.chain) % N_VAL
    assert len(ex.chain) == ex.depth
    standing = ex.start
    for value in ex.chain:
        fresh = ex.corpus.fresh(standing)
        assert fresh.val == value
        standing = fresh.nxt
    assert ex.records == tuple(
        [ex.start] + [ex.corpus.fresh(r).nxt for r in ex.records[:-1]])


def test_truth_and_the_example_agree_because_they_are_written_separately():
    """`truth()` walks the ledger; `Example.chain` is what the generator recorded."""
    for seed in range(6):
        ex = make_example(random.Random(seed), MIN_DEPTH)
        answer, chain, _end = truth(ex.corpus, ex.start, ex.depth)
        assert (answer, chain) == (ex.answer, ex.chain)


@pytest.mark.parametrize("seed", range(8))
def test_a_freshest_line_never_points_at_its_own_record(seed):
    """A self-loop would repeat one value down the chain until it could be guessed."""
    corpus = make_corpus(random.Random(seed))
    for rec in range(N_REC):
        assert corpus.fresh(rec).nxt != rec


def test_stale_versions_are_stale_and_the_freshest_is_last():
    corpus = make_corpus(random.Random(3))
    for rec in range(N_REC):
        versions = corpus.versions(rec)
        assert versions
        assert [v.ver for v in versions] == sorted(v.ver for v in versions)
        assert versions[-1] == corpus.fresh(rec)
        assert all(v.rec == rec for v in versions)


def test_two_queries_never_share_a_ledger():
    """The memorisation shortcut has to be unavailable, not merely discouraged."""
    exs = make_dataset(40, 0, EVAL_DEPTHS)
    fingerprints = {tuple(corpus.fresh(r).val for r in range(N_REC)) for corpus in
                    (e.corpus for e in exs)}
    assert len(fingerprints) == len(exs)


def test_the_question_names_only_the_start_and_the_depth():
    ex = make_example(random.Random(7), max(EVAL_DEPTHS))
    assert ex.question == f"{RECS[ex.start]} {DEPTHS[ex.depth - MIN_DEPTH]}"
    assert len(ex.question.split()) == 2


def test_the_vocabulary_is_unique_and_line_tokens_round_trip():
    assert len(set(TOK)) == N_TOK == len(I2T)
    line = make_corpus(random.Random(1)).lines[0]
    assert dec(list(line.tokens())) == " ".join(
        [RECS[line.rec], VALS[line.val], RECS[line.nxt], "[SEP]"])
    assert tok(RECS[line.rec]) == list(TOK).index(RECS[line.rec])


def test_every_depth_the_study_evaluates_is_expressible():
    """The depth token is in the question and the hop count is in the citation, so a
    depth outside the vocabulary would make a results row unanswerable by construction."""
    assert set(EVAL_DEPTHS) <= set(range(MIN_DEPTH, MIN_DEPTH + len(DEPTHS)))
    assert max(EVAL_DEPTHS) < N_CITE


def test_the_dataset_cycles_every_depth_the_study_evaluates():
    exs = make_dataset(30, 1, EVAL_DEPTHS)
    assert sorted({e.depth for e in exs}) == sorted(EVAL_DEPTHS)
    counts = {d: sum(1 for e in exs if e.depth == d) for d in EVAL_DEPTHS}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_same_seed_rebuilds_the_same_queries():
    a = [(e.start, e.depth, e.answer) for e in make_dataset(12, 4, EVAL_DEPTHS)]
    b = [(e.start, e.depth, e.answer) for e in make_dataset(12, 4, EVAL_DEPTHS)]
    assert a == b


def test_two_seeds_share_no_ledger():
    """The unit of replication is the seed, and a seed changes the facts too: two
    seeds that saw one common ledger could share a lucky memorisation."""
    a = {e.corpus for e in make_dataset(12, 0, EVAL_DEPTHS)}
    b = {e.corpus for e in make_dataset(12, 1, EVAL_DEPTHS)}
    assert len(a) == 12 and len(b) == 12
    assert a.isdisjoint(b)
