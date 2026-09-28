"""The harnesses: six ways of wiring the same frozen policy into a research loop.

Every arm runs the same two roles -- a researcher that searches and writes notes, a
reporter that folds the notes into an answer -- and they differ only in what is
allowed to travel between them. `budget` is counted in *tokens the reporter gets to
read*, the one currency the arms share: a typed note costs a token, the same note
with its provenance costs two, and a raw search result costs four.

The file keeps three failures apart that a single end-to-end number blurs:

* the researcher went to the wrong record (`walked_ok`),
* it read the right record and wrote the wrong value (`wrong_value`),
* it wrote the right value and the relay never carried it (`lost`).

Only the last one is the interface's fault, and it is the only one the reporter has
no way to notice from the inside.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from relay.kb import DEPTHS, N_VAL, NONE, RECS, SEP, VALS, Example, Line, tok
from relay.model import MAXLEN, Logits, Relay

ARMS = ("typed_relay", "cited_relay", "full_context", "plan_first", "single_shot",
        "oracle_relay")
NOTE_ARMS = ("typed_relay", "cited_relay", "plan_first", "oracle_relay")
# What one fact costs the reporter, in tokens it has to read.
COST = {"typed_relay": 1, "cited_relay": 2, "full_context": 4, "plan_first": 1,
        "single_shot": 0, "oracle_relay": 1}
DEFAULT_BUDGET = {"typed_relay": 3, "cited_relay": 6, "full_context": 36,
                  "plan_first": 3, "single_shot": 0, "oracle_relay": 3}
BUDGETS = {"typed_relay": (1, 2, 3, 4),
           "cited_relay": (2, 4, 6, 8),
           "full_context": (4, 12, 24, 36),
           "plan_first": (1, 2, 3, 4),
           "single_shot": (0,),
           "oracle_relay": (1, 2, 3, 4)}
DISTRACTORS = 2
QUERY_ROLE, REPORT_ROLE = 0, 1
HEAD_PREFIX = 4  # role, start, depth, SEP
MAX_PAYLOAD = MAXLEN - HEAD_PREFIX


@dataclass
class Trace:
    """One finished research job, with everything the metrics need to blame a stage.

    `oks` is the researcher's per-hop verdict on its own reading (did the value it
    wrote match the freshest line), kept per hop rather than pooled so a drift along
    the chain is visible. `wanted`/`carried` count the facts the report owes against
    the facts the interface actually delivered.
    """

    arm: str
    budget: int
    depth: int
    answer: int
    cite: int
    tokens: int
    visited: tuple[int, ...] = ()
    notes: tuple[int | None, ...] = ()
    oks: tuple[bool, ...] = ()
    wanted: int = 0
    carried: int = 0
    walked_ok: bool = False
    wrong_value: int = 0

    @property
    def written(self) -> int:
        return sum(1 for n in self.notes if n is not None)

    @property
    def lost(self) -> int:
        return max(0, self.wanted - self.carried)

    @property
    def partial_sum(self) -> bool:
        """Right arithmetic over a shortened chain: the answer is a prefix sum.

        Only defined for the notebook arms -- without a notebook there is no
        "shortened chain" to sum.
        """
        if not self.notes or self.carried >= self.depth:
            return False
        return self.answer == sum(v for v in self.notes if v is not None) % N_VAL


@dataclass
class _Step:
    """One researcher turn: what it wrote, and where it decided to look next.

    `fresh_val` is carried alongside rather than read off `lines`, because a search
    returns its hits in an arbitrary order -- the freshest version is a property of
    the record, not of the position it happened to land in.
    """

    rec: int
    val: int
    nxt: int
    lines: tuple[Line, ...]
    fresh_val: int

    @property
    def ok_value(self) -> bool:
        return self.val == self.fresh_val


def look(ex: Example, rec: int, salt: int) -> tuple[Line, ...]:
    """Everything a search returns for one record, in a shuffled order.

    The index resolves revisions: the hit you asked for is the *freshest* line, and
    the rest are unrelated records that came back with it. Which revision wins is
    the search engine's job here, not the researcher's, so that the only thing the
    notebook can blame a wrong answer on is the notebook.

    Deterministic in (record, salt) so a rollout and the test over it see the same
    evidence, while `salt` keeps the return order out of the answer.
    """
    rng = random.Random(f"{rec}:{salt}:{DISTRACTORS}")
    cand = [ex.corpus.fresh(rec)]
    for _ in range(DISTRACTORS):
        cand.append(ex.corpus.fresh(rng.randrange(len(RECS))))
    rng.shuffle(cand)
    return tuple(cand)


def encode(role: int, focus: int, ex: Example, payload: list[int]) -> list[int]:
    ids = [tok(("[Q]", "[R]")[role]), tok(RECS[focus]),
           tok(DEPTHS[ex.depth - 2]), tok(SEP)]
    return (ids + payload)[:MAXLEN]


def batch(model: Relay, rows: list[list[int]]) -> Logits:
    width = max(len(r) for r in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = True
    return model(ids, mask)


def lines_payload(lines: list[Line] | tuple[Line, ...]) -> list[int]:
    return [t for line in lines for t in line.tokens()]


def research(model: Relay, ex: Example, hops: int) -> list[_Step]:
    """Follow the chain with the policy's own pointer choices, for `hops` turns."""
    steps: list[_Step] = []
    rec = ex.start
    for i in range(hops):
        lines = look(ex, rec, i)
        out = batch(model, [encode(QUERY_ROLE, rec, ex, lines_payload(lines))])
        step = _Step(rec=rec, val=int(out.note.argmax(-1)[0]),
                     nxt=int(out.nxt.argmax(-1)[0]), lines=lines,
                     fresh_val=ex.corpus.fresh(rec).val)
        steps.append(step)
        rec = step.nxt
    return steps


def notebook(notes: list[int | None], visited: list[int], cost: int) -> list[int]:
    payload: list[int] = []
    for i, value in enumerate(notes):
        payload.append(tok(VALS[value]) if value is not None else tok(NONE))
        if cost == 2:
            rec = visited[i] if i < len(visited) and value is not None else None
            payload.append(tok(RECS[rec]) if rec is not None else tok(NONE))
    return payload


def chain_records(ex: Example) -> list[int]:
    recs = [ex.start]
    for _ in range(ex.depth - 1):
        recs.append(ex.corpus.fresh(recs[-1]).nxt)
    return recs


def _visible_fresh(payload: list[int], ex: Example, visited: list[int]) -> int:
    """How many hops are still fully recoverable from a clipped evidence window."""
    lines = [tuple(payload[i:i + 4]) for i in range(0, len(payload) - 3, 4)]
    held = set()
    for hop, rec in enumerate(visited):
        fresh = ex.corpus.fresh(rec)
        if fresh.tokens() in lines:
            held.add((hop, fresh.val))
    return sum(1 for hop in range(len(visited)) if any(h == hop for h, _ in held))


def run(model: Relay, ex: Example, arm: str, budget: int | None = None) -> Trace:
    """One research job end to end, with the bookkeeping the metrics need."""
    budget = DEFAULT_BUDGET[arm] if budget is None else budget
    capacity = budget // COST[arm] if COST[arm] else 0
    truth_records = chain_records(ex)

    if arm == "single_shot":
        out = batch(model, [encode(REPORT_ROLE, ex.start, ex, [])])
        return Trace(arm=arm, budget=budget, depth=ex.depth,
                     answer=int(out.ans.argmax(-1)[0]), cite=int(out.cite.argmax(-1)[0]),
                     tokens=HEAD_PREFIX, wanted=ex.depth)

    steps = research(model, ex, hops=1 if arm == "plan_first" else ex.depth)
    visited = [s.rec for s in steps]
    oks = tuple(s.ok_value for s in steps)
    if arm == "plan_first":
        visited = [ex.start] + [steps[-1].nxt] * (ex.depth - 1)
    walked_ok = visited == truth_records

    if arm == "full_context":
        payload = lines_payload([line for i, rec in enumerate(visited)
                                  for line in look(ex, rec, i)])
        payload = payload[:min(budget, MAX_PAYLOAD)]
        out = batch(model, [encode(REPORT_ROLE, ex.start, ex, payload)])
        held = _visible_fresh(payload, ex, visited)
        return Trace(arm=arm, budget=budget, depth=ex.depth,
                     answer=int(out.ans.argmax(-1)[0]),
                     cite=int(out.cite.argmax(-1)[0]),
                     tokens=HEAD_PREFIX + len(payload), visited=tuple(visited),
                     oks=oks,
                     wanted=len(visited), carried=held,
                     walked_ok=walked_ok)

    notes: list[int | None] = [None] * capacity
    wanted, wrong = len(steps), 0
    for i, step in enumerate(steps):
        wrong += int(not step.ok_value)
        value = ex.chain[i] if arm == "oracle_relay" else step.val
        if i < capacity:
            notes[i] = value
    if arm == "plan_first":
        wanted = ex.depth  # the report still owes `depth` facts; one search cannot pay them
    payload = notebook(notes, visited, COST[arm])[:MAX_PAYLOAD]
    out = batch(model, [encode(REPORT_ROLE, ex.start, ex, payload)])
    carried = sum(1 for n in notes if n is not None)
    return Trace(arm=arm, budget=budget, depth=ex.depth,
                 answer=int(out.ans.argmax(-1)[0]), cite=int(out.cite.argmax(-1)[0]),
                 tokens=HEAD_PREFIX + len(payload), visited=tuple(visited),
                 oks=oks, notes=tuple(notes), wanted=wanted, carried=carried,
                 walked_ok=walked_ok, wrong_value=wrong)


def run_many(model: Relay, exs: list[Example], arm: str,
             budget: int | None = None) -> list[Trace]:
    """The same rollout, batched across queries: one forward pass per hop.

    Kept deliberately parallel to `run` rather than sharing its body, and a test
    checks the two agree on the same examples -- a batched shortcut that quietly
    changes argmaxes would move every number in the README.
    """
    budget = DEFAULT_BUDGET[arm] if budget is None else budget
    cost = COST[arm]
    capacity = budget // cost if cost else 0
    device_notes: list[list[int | None]] = [[None] * capacity for _ in exs]
    visited: list[list[int]] = [[ex.start] for ex in exs]
    wanted = [0] * len(exs)
    wrong = [0] * len(exs)

    if arm == "single_shot":
        out = batch(model, [encode(REPORT_ROLE, ex.start, ex, []) for ex in exs])
        return [Trace(arm=arm, budget=budget, depth=ex.depth,
                      answer=int(out.ans.argmax(-1)[i]),
                      cite=int(out.cite.argmax(-1)[i]), tokens=HEAD_PREFIX,
                      wanted=ex.depth) for i, ex in enumerate(exs)]

    hops = 1 if arm == "plan_first" else max(ex.depth for ex in exs)
    vals: list[list[int]] = [[] for _ in exs]
    nexts: list[list[int]] = [[] for _ in exs]
    oks: list[list[bool]] = [[] for _ in exs]
    for hop in range(hops):
        rows, live = [], []
        for i, ex in enumerate(exs):
            if hop >= ex.depth:
                continue
            rec = visited[i][-1]
            rows.append(encode(QUERY_ROLE, rec, ex,
                               lines_payload(look(ex, rec, hop))))
            live.append(i)
        out = batch(model, rows)
        note, nxt = out.note.argmax(-1), out.nxt.argmax(-1)
        for slot, i in enumerate(live):
            ex = exs[i]
            vals[i].append(int(note[slot]))
            nexts[i].append(int(nxt[slot]))
            wanted[i] += 1
            fresh = ex.corpus.fresh(visited[i][-1])
            ok = vals[i][-1] == fresh.val
            oks[i].append(ok)
            wrong[i] += int(not ok)
            if hop + 1 < ex.depth:
                visited[i].append(nexts[i][-1])

    traces: list[Trace] = []
    for i, ex in enumerate(exs):
        if arm == "plan_first":
            visited[i] = [ex.start] + [nexts[i][0]] * (ex.depth - 1)
            wanted[i] = ex.depth
        truth_records = chain_records(ex)
        walked = visited[i] == truth_records
        if arm == "full_context":
            payload = lines_payload([line for h, rec in enumerate(visited[i])
                                      for line in look(ex, rec, h)])
            payload = payload[:min(budget, MAX_PAYLOAD)]
            held = _visible_fresh(payload, ex, visited[i])
            row = [encode(REPORT_ROLE, ex.start, ex, payload)]
            out = batch(model, row)
            traces.append(Trace(arm=arm, budget=budget, depth=ex.depth,
                                answer=int(out.ans.argmax(-1)[0]),
                                cite=int(out.cite.argmax(-1)[0]),
                                tokens=HEAD_PREFIX + len(payload),
                                visited=tuple(visited[i]), oks=tuple(oks[i]),
                                wanted=len(visited[i]),
                                carried=held, walked_ok=walked))
            continue
        for h in range(len(vals[i])):
            value = ex.chain[h] if arm == "oracle_relay" else vals[i][h]
            if h < capacity:
                device_notes[i][h] = value
        payload = notebook(device_notes[i], visited[i], cost)[:MAX_PAYLOAD]
        out = batch(model, [encode(REPORT_ROLE, ex.start, ex, payload)])
        traces.append(Trace(arm=arm, budget=budget, depth=ex.depth,
                            answer=int(out.ans.argmax(-1)[0]),
                            cite=int(out.cite.argmax(-1)[0]),
                            tokens=HEAD_PREFIX + len(payload),
                            visited=tuple(visited[i]), oks=tuple(oks[i]),
                            notes=tuple(device_notes[i]),
                            wanted=wanted[i],
                            carried=sum(1 for n in device_notes[i] if n is not None),
                            walked_ok=walked, wrong_value=wrong[i]))
    return traces


def render_notebook(trace: Trace) -> str:
    """The notebook as a reader sees it, for the CLI."""
    if not trace.notes:
        return "(no notebook: the reporter read raw evidence)"
    return " ".join(VALS[v] if v is not None else NONE for v in trace.notes)
