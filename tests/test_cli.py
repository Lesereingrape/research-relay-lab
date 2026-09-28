"""Every command printed in the README must run, and print what it claims to.

The Quickstart is the part of a README a reader actually executes. The two model-free
commands are pinned here byte for byte, because their output is arithmetic and cannot
depend on a float reduction order -- if the CLI's bookkeeping and the harness's ever
disagree, that is the bug the pinned transcript exists to catch.

The demo is *not* byte-pinned: it trains, and a different thread count can flip one
near-tie argmax on another machine. What it is held to instead is that it runs, that
every arm and both frontiers appear in the order the study reports them, that an
unmeasurable condition prints `n/a` rather than a plausible zero, and that it learns
enough to separate the arms.
"""

from __future__ import annotations

import contextlib
import io
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from relay.cli import DEMO_EVAL, DEMO_STEPS, DEMO_TRAIN, FRONTIER, main
from relay.harness import (
    ARMS,
    COST,
    NOTE_ARMS,
    _visible_fresh,
    lines_payload,
    look,
    notebook,
)
from relay.kb import N_VAL, VALS, dec, make_example, truth
from relay.model import count_parameters
from relay.study import EVAL_DEPTHS, N_EVAL, N_TRAIN, SEEDS
from relay.train import STEPS

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```\n(.*?)\n```", re.DOTALL)


def _readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _quickstart() -> str:
    return _readme().split("## Quickstart")[1].split("## Design")[0]


def _commands() -> list[tuple[str, str | None]]:
    """`(command, expected output or None)` for every fenced command, in order."""
    out: list[tuple[str, str | None]] = []
    for body in FENCE.findall(_quickstart()):
        lines = body.splitlines()
        if lines and all(ln.startswith("$ ") for ln in lines):
            out.extend((ln[2:], None) for ln in lines)
        elif out and out[-1][1] is None and not out[-1][0].startswith("python "):
            out[-1] = (out[-1][0], body)
    return out


def _pinned() -> list[tuple[str, str]]:
    return [(c, o) for c, o in _commands()
            if o is not None and not c.startswith("relay demo")]


def _demo_transcript() -> str:
    """The README's copy of the demo's output, if it prints one.

    It is not byte-pinned -- the command trains -- but its *shape* is checkable, and a
    transcript whose header still claims 800 queries or whose arms are in a different
    order than the code reports them is a transcript nobody ran.
    """
    for cmd, output in _commands():
        if cmd.startswith("relay demo"):
            assert output is not None, "the README shows the demo command with no output"
            return output
    raise AssertionError("the Quickstart does not print a demo transcript")


def _capture(argv: list[str]) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(argv) == 0
    return buf.getvalue()


def test_the_quickstart_documents_the_commands_the_package_ships():
    cmds = [c for c, _ in _commands()]
    assert any(c.startswith("relay ledger") for c in cmds)
    assert any(c.startswith("relay notebook") for c in cmds)
    assert "relay demo" in cmds
    assert cmds[-2:] == ["python experiments/run_study.py",
                         "python experiments/make_report.py --write"]


def test_every_pinned_transcript_belongs_to_a_model_free_command():
    """The demo trains, so its digits are machine-dependent and must not be pinned."""
    for cmd, _ in _pinned():
        assert cmd.startswith("relay ledger") or cmd.startswith("relay notebook")


def test_the_readme_demo_transcript_has_the_shape_the_command_prints():
    """The transcript is not byte-pinned, but it is not allowed to be a story either."""
    lines = _demo_transcript().strip().splitlines()
    assert len(lines) == 2 + len(ARMS) + len(FRONTIER), lines
    for claim in (f"{DEMO_TRAIN} train", f"{DEMO_EVAL} held-out queries",
                  f"{DEMO_STEPS} steps", f"{count_parameters():,} params"):
        assert claim in lines[0], (
            f"the README shows a split the code does not run: {claim}")
    assert lines[1].startswith("shared policy")
    for arm, line in zip(ARMS, lines[2:2 + len(ARMS)], strict=True):
        assert line.startswith(arm), line
        for token in ("acc", "walked", "misread", "lost", "over", "tok"):
            assert token in line, line
    for arm, line in zip(FRONTIER, lines[2 + len(ARMS):], strict=True):
        assert line.startswith(f"{arm} frontier"), line


@pytest.mark.parametrize("cmd, expected", _pinned(), ids=[c for c, _ in _pinned()])
def test_a_pinned_transcript_is_what_the_command_prints(cmd, expected):
    """The block under a Quickstart command is regenerated and compared byte for byte."""
    assert _capture(cmd.split()[1:]).strip("\n") == expected.strip("\n")


# --- ledger ------------------------------------------------------------------

def test_the_ledger_command_agrees_with_the_verifier_it_advertises():
    text = _capture(["ledger", "--seed", "5", "--depth", "3"])
    ex = make_example(random.Random(5), 3)
    answer, _chain, _end = truth(ex.corpus, ex.start, 3)
    assert f"query: {ex.question}" in text
    assert f"answer: {VALS[answer]}" in text
    assert text.rstrip().endswith(f"({' + '.join(VALS[v] for v in ex.chain)} mod "
                                  f"{N_VAL})")


def test_the_ledger_shows_the_stale_versions_a_search_has_to_see_through():
    text = _capture(["ledger", "--seed", "5", "--depth", "3"])
    assert "stale line(s)" in text
    assert "freshest = " in text
    assert len([ln for ln in text.splitlines() if ln.startswith("hop ")]) == 3


def test_the_ledgers_running_sum_is_the_verifiers_fold():
    """Each hop's stated running total has to be the prefix sum mod the value count."""
    ex = make_example(random.Random(11), 3)
    text = _capture(["ledger", "--seed", "11", "--depth", "3"])
    totals = [int(re.search(r"running sum v(\d)", ln).group(1))
              for ln in text.splitlines() if "running sum" in ln]
    assert totals == [sum(ex.chain[:i + 1]) % N_VAL for i in range(3)]


# --- notebook ----------------------------------------------------------------

def test_the_notebook_command_counts_the_budget_the_way_the_harness_does():
    text = _capture(["notebook", "--arm", "cited_relay", "--budget", "4", "--depth", "3"])
    assert "so 2 of 3 hops fit" in text
    assert "carried 2, lost 1" in text
    assert "head 4 + payload 4 = 8 tokens the reporter reads" in text
    assert "cites more than the interface delivered" in text


def test_a_budget_the_chain_fits_reports_no_loss():
    text = _capture(["notebook", "--arm", "typed_relay", "--budget", "3", "--depth", "3"])
    assert "so 3 of 3 hops fit" in text
    assert "carried 3, lost 0" in text
    assert "every hop survived" in text


def test_the_notebook_payload_is_the_harness_function_not_a_reprint():
    """The CLI has to render what `notebook()` builds, or the transcript is a story."""
    ex = make_example(random.Random(5), 3)
    for arm, budget in (("typed_relay", 3), ("cited_relay", 4)):
        cost = COST[arm]
        payload = notebook(list(ex.chain[:budget // cost]), list(ex.records), cost)
        text = _capture(["notebook", "--arm", arm, "--budget", str(budget),
                         "--seed", "5", "--depth", "3"])
        assert f"report payload: {dec(payload)}" in text
        assert f"head 4 + payload {len(payload)}" in text


def test_the_notebook_folds_the_surviving_notes_exactly():
    ex = make_example(random.Random(5), 3)
    text = _capture(["notebook", "--arm", "typed_relay", "--budget", "2", "--seed", "5",
                     "--depth", "3"])
    surviving = list(ex.chain[:2])
    assert (f"folding the 2 surviving notes gives "
            f"{VALS[sum(surviving) % N_VAL]} against a true {VALS[ex.answer]}" in text)


def test_the_raw_hit_arm_counts_hops_present_rather_than_lines_that_fit():
    """Four tokens is one line and no hop: the freshest line of a hop is one of three."""
    ex = make_example(random.Random(5), 3)
    payload = lines_payload([line for h, rec in enumerate(ex.records)
                             for line in look(ex, rec, h)])[:4]
    held = _visible_fresh(payload, ex, list(ex.records))
    text = _capture(["notebook", "--arm", "full_context", "--budget", "4", "--seed", "5",
                     "--depth", "3"])
    assert f"so {held} of 3 hops fit" in text
    assert f"head 4 + payload {len(payload)} = {4 + len(payload)}" in text


def test_only_the_arms_with_a_notebook_are_described_as_having_one():
    """`single_shot` has no window, so it must not be reported as an empty one."""
    for arm in ARMS:
        text = _capture(["notebook", "--arm", arm, "--budget", "1", "--depth", "2"])
        assert ("nothing travels between the query and the report" in text) == \
            (arm == "single_shot")
        assert ("surviving notes" in text) == (arm in NOTE_ARMS)


@pytest.mark.parametrize("budget", [9, 100])
def test_the_notebook_command_refuses_a_budget_that_cannot_truncate(budget):
    """The command exists to show a loss; a budget that loses nothing is a mistake."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), pytest.raises(SystemExit) as exit_code:
        main(["notebook", "--budget", str(budget), "--depth", "2"])
    assert exit_code.value.code == 2
    assert "more than 2 hops cost" in buf.getvalue()


def test_the_raw_hit_arm_allows_its_own_default_budget():
    """A window sized to hold every line is still the interesting case for `full_context`."""
    text = _capture(["notebook", "--arm", "full_context", "--budget", "24", "--depth",
                     "2"])
    assert "hops fit" in text


def test_an_unknown_arm_is_rejected_rather_than_guessed():
    with pytest.raises(SystemExit):
        _capture(["notebook", "--arm", "summarizer"])


@pytest.mark.parametrize("depth", [min(EVAL_DEPTHS) - 1, max(EVAL_DEPTHS) + 1])
def test_a_depth_outside_the_study_is_rejected_rather_than_clamped(depth):
    with pytest.raises(SystemExit):
        _capture(["ledger", "--depth", str(depth)])


# --- demo --------------------------------------------------------------------

def test_demo_is_a_smaller_run_than_the_study():
    """The quickstart budgets minutes; the README's tables come from a bigger run."""
    assert DEMO_TRAIN < N_TRAIN and DEMO_EVAL < N_EVAL and DEMO_STEPS < STEPS
    assert SEEDS[0] == 0


@pytest.fixture(scope="module")
def demo_lines() -> list[str]:
    """One trained demo per module: it costs minutes, and every check reads the same text."""
    return _capture(["demo", "--seed", "0"]).strip().splitlines()


def test_demo_prints_every_arm_and_the_shared_policy(demo_lines):
    arms = demo_lines[2:2 + len(ARMS)]
    assert len(demo_lines) == 2 + len(ARMS) + len(FRONTIER)
    assert demo_lines[0].startswith("seed 0:")
    assert "params" in demo_lines[0] and "CPU" in demo_lines[0]
    assert demo_lines[1].startswith("shared policy")
    for arm, line in zip(ARMS, arms, strict=True):
        assert line.startswith(arm), line
        for token in ("acc", "walked", "misread", "lost", "over", "tok"):
            assert token in line
    for line in demo_lines[2 + len(ARMS):]:
        assert line.startswith(tuple(f"{a} frontier" for a in FRONTIER))


def test_demo_reports_a_question_an_arm_cannot_answer_as_not_applicable(demo_lines):
    """`single_shot` never walks, so its walk accuracy is not low, it is undefined."""
    single = next(ln for ln in demo_lines if ln.startswith("single_shot"))
    raw = next(ln for ln in demo_lines if ln.startswith("full_context"))
    assert "walked   n/a" in single and "misread   n/a" in single
    assert "misread   n/a" in raw and "walked   n/a" not in raw


def test_demo_actually_learns_something(demo_lines):
    """A transcript of an untrained policy would still print a plausible table."""
    accs = {ln.split()[0]: float(re.search(r"acc ([\d.]+)", ln).group(1))
            for ln in demo_lines[2:2 + len(ARMS)]}
    assert accs["oracle_relay"] > 0.5, demo_lines
    assert accs["typed_relay"] > accs["single_shot"], demo_lines
    assert accs["single_shot"] < 0.35, demo_lines
    assert all(0.0 <= a <= 1.0 for a in accs.values())


def test_a_short_demo_run_is_deterministic_in_this_environment():
    """Same process, same seed, same output -- the numbers only travel across machines."""
    assert _capture(["demo", "--seed", "1", "--steps", "30"]) == \
        _capture(["demo", "--seed", "1", "--steps", "30"])


def test_the_study_script_advertises_its_scratch_and_compare_flags():
    """The escape hatches the README advertises have to be the ones that exist."""
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "run_study.py"), "--help"],
        capture_output=True, text=True, check=True, cwd=ROOT)
    assert "--out" in help_text.stdout and "--against" in help_text.stdout
