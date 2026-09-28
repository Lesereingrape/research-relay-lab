"""The trunk is one small model, and every arm shares it.

Two properties matter and neither is visible in a results table: the padding mask has
to be real (a row that gets padded to the batch width must not change its answer, or
the number of facts a reporter was handed would leak in through the zeros), and the
readout has to be a sum over positions (a mean would let the notebook's *length* show
up in the magnitude of every value in it).
"""

from __future__ import annotations

import pytest
import torch

from relay.kb import N_CITE, N_REC, N_VAL, SEP, tok
from relay.model import (
    D_MODEL,
    DROPOUT,
    FF,
    HEADS,
    LAYERS,
    MAXLEN,
    Relay,
    build,
    count_parameters,
)


@pytest.fixture(scope="module")
def model() -> Relay:
    return build(0)


def test_the_policy_is_small_enough_to_be_an_explanation():
    """The README calls this a laptop-sized model; that is a claim about a number."""
    n = count_parameters()
    assert 50_000 < n < 250_000


def test_an_arm_cannot_smuggle_in_extra_weights(model):
    """There is exactly one set of weights, so a difference between arms is a
    difference in the harness rather than in what was trained."""
    heads = {name: getattr(model, name) for name in ("note", "nxt", "ans", "cite")}
    assert all(isinstance(h, torch.nn.Linear) for h in heads.values())
    assert {h.out_features for h in heads.values()} == {N_VAL, N_REC, N_CITE}
    assert all(h.in_features == D_MODEL for h in heads.values())
    assert len(model.trunk.layers) == LAYERS


def test_the_head_classes_answer_to_the_vocabulary(model):
    ids = torch.tensor([[tok(SEP)] * 5])
    out = model.forward(ids, torch.ones(1, 5, dtype=torch.bool))
    assert (out.note.shape[1], out.ans.shape[1]) == (N_VAL, N_VAL)
    assert (out.nxt.shape[1], out.cite.shape[1]) == (N_REC, N_CITE)


def test_padding_a_row_does_not_change_what_it_says(model):
    row = [tok(SEP), 3, 7, 11]
    live = model.forward(torch.tensor([row]), torch.ones(1, len(row), dtype=torch.bool))
    padded = model.forward(
        torch.tensor([row + [0] * 6]),
        torch.tensor([[True] * len(row) + [False] * 6]))
    for name in ("note", "nxt", "ans", "cite"):
        assert torch.allclose(getattr(live, name), getattr(padded, name), atol=1e-5), name


def test_the_sum_readout_lets_a_longer_notebook_add_up_rather_than_dilute(model):
    """Two facts about value 1 should read differently from one, which a mean-pooled
    readout would paper over by dividing by the number of positions."""
    one = [tok(SEP), tok(SEP)]
    two = [tok(SEP), tok(SEP), tok(SEP), tok(SEP)]
    a = model(torch.tensor([one]), torch.ones(1, 2, dtype=torch.bool)).ans
    b = model(torch.tensor([two]), torch.ones(1, 4, dtype=torch.bool)).ans
    assert not torch.allclose(a, b, atol=1e-3)


def test_an_uninitialised_position_cannot_be_reached(model):
    ids = torch.arange(MAXLEN).tolist()
    out = model.forward(torch.tensor([ids]), torch.ones(1, MAXLEN, dtype=torch.bool))
    assert torch.isfinite(out.note).all()


def test_the_same_seed_builds_the_same_policy():
    a, b = build(7), build(7)
    assert torch.equal(a.embed.weight, b.embed.weight)
    assert not torch.equal(build(7).embed.weight, build(8).embed.weight)


def test_no_arm_gets_a_dropout_escape_hatch():
    """A stochastic harness would make a per-seed spread mean something else."""
    assert DROPOUT == 0.0
    assert FF > D_MODEL and HEADS <= D_MODEL
