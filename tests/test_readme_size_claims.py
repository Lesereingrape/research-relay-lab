"""The README's non-result numbers: every size, count and shape it spells out.

Result figures are byte-pinned to the artifact elsewhere. This file covers the claims
around them -- how many records the ledger has, what the modulo is, how wide the
transformer is, how many seeds ran -- because those are copied out of the code by hand,
and a refactor that changed one would otherwise leave a confident sentence standing on
a number nothing in the repo produces any more.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path

import relay.harness as harness
import relay.kb as kb
import relay.model as model
import relay.study as study
from relay.train import BATCH, LR, STEPS

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
ART = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
CFG = ART["config"]
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _says(text: str) -> None:
    """Case-insensitive: a sentence-initial capital is not a changed claim."""
    assert re.search(re.escape(text), README, re.IGNORECASE), \
        f"README no longer says {text!r}"


def test_the_headline_parameter_count_is_the_model_the_code_builds():
    claim = re.search(r"(\d+)k-parameter", README)
    assert claim, "the README's parameter claim has changed shape"
    assert model.count_parameters() == CFG["params"]
    assert int(claim.group(1)) == round(CFG["params"] / 1000)


def test_the_prose_describes_the_architecture_the_module_declares():
    _says(f"{model.D_MODEL}-wide, {model.HEADS}-head, {model.LAYERS}-layer")
    heads = [f.name for f in fields(model.Logits)]
    _says(f"{WORDS[len(heads)]} typed heads")
    assert heads == ["note", "nxt", "ans", "cite"]
    assert "sum-pooled" in README


def test_the_task_description_matches_the_schema_the_verifier_uses():
    _says(f"A ledger of {kb.N_REC} records")
    _says(f"one to {WORDS[max(kb.VERSIONS)]} versions")
    _says(f"a depth of {WORDS[min(study.EVAL_DEPTHS)]} or "
          f"{WORDS[max(study.EVAL_DEPTHS)]} hops")
    _says(f"modulo {WORDS[kb.N_VAL]}")
    assert CFG["n_values"] == kb.N_VAL and CFG["n_records"] == kb.N_REC
    assert CFG["versions"] == list(kb.VERSIONS)
    assert sorted(CFG["eval_depths"]) == [kb.MIN_DEPTH, kb.MAX_DEPTH]


def test_the_evidence_the_prose_describes_is_the_evidence_the_harness_returns():
    _says(f"plus {WORDS[harness.DISTRACTORS]} lines from unrelated records")
    _says(f"{WORDS[1 + harness.DISTRACTORS]} lines")
    assert CFG["distractors"] == harness.DISTRACTORS


def test_the_pricing_table_prices_the_arms_the_harness_prices():
    rows: dict[str, int] = {}
    for line in README.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 2 and cells[1].isdigit():
            for arm in re.findall(r"`(\w+)`", cells[0]):
                rows[arm] = int(cells[1])
    assert set(rows) == set(harness.ARMS), f"the table names {sorted(rows)}"
    assert rows == {arm: harness.COST[arm] for arm in rows}
    assert rows == CFG["cost"]


def test_the_harness_count_and_names_are_the_ones_that_exist():
    _says(f"{WORDS[len(harness.ARMS)]} harnesses")
    _says(f"{WORDS[len(harness.ARMS)]} arms")
    assert CFG["arms"] == list(harness.ARMS)
    for arm in harness.ARMS:
        _says(f"`{arm}`")


def test_the_seeds_and_split_the_prose_names_are_the_ones_that_ran():
    assert CFG["seeds"] == list(study.SEEDS)
    assert CFG["n_train"] == study.N_TRAIN and CFG["n_eval"] == study.N_EVAL
    assert CFG["steps"] == STEPS and CFG["batch"] == BATCH and CFG["lr"] == LR
    _says(f"all {WORDS[len(study.SEEDS)]} seeds")
    _says(f"{WORDS[len(study.SEEDS)]} seeds, one budget sweep")


def test_the_quickstart_documents_commands_that_exist():
    for cmd in ("relay ledger", "relay notebook", "relay demo"):
        assert f"$ {cmd}" in README
    rerun = next((line for line in README.splitlines()
                  if line.startswith("python experiments/run_study.py ")), "")
    assert "--out" in rerun and "--against" in rerun, README


def test_no_placeholder_survives_into_the_readme():
    for stray in ("TODO", "TBD", "FIXME", "0.xxx", "placeholder"):
        assert stray not in README, stray
    assert re.search(r"<!-- RESULTS:START -->\n.", README), "the results block is empty"
