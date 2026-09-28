"""The knowledge base, the chains inside it, and the exact truth.

A record is a line `r07 v4 r12`: standing on record `r07`, its current value is 4,
and the chain continues at `r12`. A query says: stand on a record, walk `depth` hops
always taking the freshest line of the record you land on, and add up the values you
stood on modulo 10. That is computable without any model, which is the entire reason
this repo exists: every number in the README is scored against `truth()` rather than
against a preference.

Each query gets its *own* ledger. That is not a stylistic choice -- with one shared
ledger a policy can memorise "record 7 is worth 4" and answer without ever reading
the evidence it was handed, and then the study measures the memorisation instead of
the relay. Per-query ledgers make the shortcut unavailable: the rule repeats, the
facts never do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

VALS = tuple(f"v{i}" for i in range(5))
RECS = tuple(f"r{i:02d}" for i in range(64))
VERS = ("n1", "n2", "n3")
DEPTHS = ("d2", "d3")
ROLES = ("[Q]", "[R]")
CITES = tuple(f"c{i}" for i in range(4))
SEP = "[SEP]"
NONE = "[-]"

TOK = (*ROLES, *DEPTHS, *CITES, SEP, NONE, *VERS, *VALS, *RECS)
T2I = {t: i for i, t in enumerate(TOK)}
I2T = tuple(TOK)
N_TOK = len(TOK)

N_REC = len(RECS)
N_VAL = len(VALS)
N_CITE = len(CITES)
MIN_DEPTH = 2
MAX_DEPTH = MIN_DEPTH + len(DEPTHS) - 1
VERSIONS = (1, 2, 3)


def tok(s: str) -> int:
    return T2I[s]


def dec(ids: list[int] | tuple[int, ...]) -> str:
    return " ".join(I2T[i] for i in ids)


@dataclass(frozen=True)
class Line:
    """One version of one record: value `val`, successor `nxt`."""

    rec: int
    ver: int
    val: int
    nxt: int

    def tokens(self) -> tuple[int, ...]:
        return (tok(RECS[self.rec]), tok(VALS[self.val]), tok(RECS[self.nxt]),
                tok(SEP))


@dataclass(frozen=True)
class Corpus:
    """A ledger. `by_rec[i]` holds record i's lines, oldest version first."""

    lines: tuple[Line, ...]
    by_rec: tuple[tuple[Line, ...], ...]

    def versions(self, rec: int) -> tuple[Line, ...]:
        return self.by_rec[rec]

    def fresh(self, rec: int) -> Line:
        return self.by_rec[rec][-1]


def make_corpus(rng: random.Random, n_versions: tuple[int, ...] = VERSIONS) -> Corpus:
    """A ledger whose freshest lines form a chain, with stale versions alongside.

    The freshest line of a record never points at the record itself: a self-loop
    would make every deeper query land on the same value over and over, and a policy
    could pass those queries without reading anything.
    """
    lines: list[Line] = []
    for rec in range(N_REC):
        rec_lines: list[Line] = []
        for ver in range(n_versions[rng.randrange(len(n_versions))]):
            nxt = rng.randrange(N_REC)
            while nxt == rec:
                nxt = rng.randrange(N_REC)
            rec_lines.append(Line(rec=rec, ver=ver, val=rng.randrange(N_VAL), nxt=nxt))
        lines.extend(rec_lines)
    by_rec = tuple(tuple(sorted((ln for ln in lines if ln.rec == rec),
                                key=lambda ln: ln.ver)) for rec in range(N_REC))
    return Corpus(lines=tuple(lines), by_rec=by_rec)


def truth(corpus: Corpus, start: int, depth: int) -> tuple[int, tuple[int, ...], int]:
    """Fold the chain: (answer, the value of every hop, the record we ended on)."""
    vals: list[int] = []
    rec = start
    for _ in range(depth):
        line = corpus.fresh(rec)
        vals.append(line.val)
        rec = line.nxt
    return sum(vals) % N_VAL, tuple(vals), rec


@dataclass(frozen=True)
class Example:
    corpus: Corpus
    start: int
    depth: int
    answer: int
    chain: tuple[int, ...]

    @property
    def question(self) -> str:
        return f"{RECS[self.start]} {DEPTHS[self.depth - MIN_DEPTH]}"

    @property
    def records(self) -> tuple[int, ...]:
        """The records the chain actually walks through, start first."""
        recs = [self.start]
        for _ in range(self.depth - 1):
            recs.append(self.corpus.fresh(recs[-1]).nxt)
        return tuple(recs)


def make_example(rng: random.Random, depth: int) -> Example:
    corpus = make_corpus(rng)
    start = rng.randrange(N_REC)
    answer, chain, _end = truth(corpus, start, depth)
    return Example(corpus=corpus, start=start, depth=depth, answer=answer, chain=chain)


def make_dataset(n: int, seed: int, depths: tuple[int, ...]) -> list[Example]:
    """`n` queries, one fresh ledger each, depths cycled so every one is exercised.

    The seed sets the ledger stream, so two seeds see disjoint facts *and* different
    initialisations -- which is what makes the spread across seeds worth printing.
    """
    rng = random.Random(seed)
    return [make_example(rng, depths[i % len(depths)]) for i in range(n)]
