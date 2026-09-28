"""Scoring: decompose a wrong answer into the stage that owes it.

Every rate here is a mean over held-out queries, and every one of them is checkable
without the model: the chain, the notebook and the report's own citation claim are
all in the trace. `metrics` deliberately does not know what a harness *should* do --
it only asks which of the three failures happened.

A condition an arm cannot be observed under is scored ``None``, not ``0.0``. An arm
with no notebook has no "accuracy when the notebook was full"; printing 0.000 there
would read as a disastrous result rather than as the absence of a question.
"""

from __future__ import annotations

from relay.harness import NOTE_ARMS, Trace
from relay.kb import Example

FIELDS = ("answer_acc", "walked_ok", "wrong_value_rate", "lost_rate", "carried_rate",
          "cite_overclaim", "partial_sum_rate", "acc_full_notebook",
          "acc_when_walked", "reporter_tokens")
# Which arms can be asked which question. `single_shot` never walks, so it has no
# walk to be right about; only the notebook arms write a value down, so only they can
# misread one.
HAS_WALK = ("typed_relay", "cited_relay", "full_context", "plan_first", "oracle_relay")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def score(pairs: list[tuple[Example, Trace]]) -> dict:
    if not pairs:
        return dict.fromkeys(FIELDS)
    hits, walks, wrongs, losts, carries, claims, tokens = [], [], [], [], [], [], []
    partials: list[float] = []
    cond_full: list[float] = []
    cond_walked: list[float] = []
    for ex, tr in pairs:
        right = float(tr.answer == ex.answer)
        notebook = tr.arm in NOTE_ARMS
        hits.append(right)
        losts.append(tr.lost / max(1, ex.depth))
        carries.append(tr.carried / max(1, ex.depth))
        claims.append(float(tr.cite > tr.carried))
        tokens.append(float(tr.tokens))
        if tr.arm in HAS_WALK:
            walks.append(float(tr.walked_ok))
            if tr.walked_ok:
                cond_walked.append(right)
        if notebook:
            wrongs.append(tr.wrong_value / max(1, tr.wanted))
            partials.append(float(tr.partial_sum))
            if tr.carried >= ex.depth:
                cond_full.append(right)
    out = {
        "answer_acc": _round(_mean(hits), 4),
        "walked_ok": _round(_mean(walks), 4),
        "wrong_value_rate": _round(_mean(wrongs), 4),
        "lost_rate": _round(_mean(losts), 4),
        "carried_rate": _round(_mean(carries), 4),
        "cite_overclaim": _round(_mean(claims), 4),
        "partial_sum_rate": _round(_mean(partials), 4),
        "acc_full_notebook": _round(_mean(cond_full), 4),
        "acc_when_walked": _round(_mean(cond_walked), 4),
        "reporter_tokens": _round(_mean(tokens), 1),
    }
    assert tuple(out) == FIELDS
    return out


def by_field(field: str, pairs: list[tuple[Example, Trace]]):
    """One number out of the same pile, for the CLI and the tests."""
    return score(pairs)[field]


def hop_profile(pairs: list[tuple[Example, Trace]]) -> list:
    """How often the researcher misreads the record it stands on, by hop index.

    Hop 0 is the record the query names, so a flat profile means the errors belong to
    the reading task, while a rising one means the walk drifts further from anything
    it can check. A hop no query reached is ``None``, not 0.0: there is no rate there.
    """
    hops = max((len(tr.oks) for _, tr in pairs), default=0)
    if not hops:
        return []
    seen = [0] * hops
    errs = [0] * hops
    for _ex, tr in pairs:
        for i, ok in enumerate(tr.oks):
            seen[i] += 1
            errs[i] += int(not ok)
    return [None if not n else round(e / n, 4) for e, n in zip(errs, seen, strict=True)]
