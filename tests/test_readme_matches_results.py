"""The README's results block is byte-pinned to the committed artifact.

Nothing between the RESULTS markers is hand-copied: the test re-runs the renderer over
``results/frontier.json`` and compares the text. A rerun of the study that changes a
number changes the README, or this fails.

The block is also the only place the README may state a result. A ``0.955`` typed into
the Design section is a claim nothing rechecks, so the digit pattern is scanned for
outside the markers.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _load_renderer():
    path = ROOT / "experiments" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block(text: str) -> str:
    m = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->", text,
                  re.DOTALL)
    assert m, "README is missing the RESULTS:START/END block"
    return m.group(1).strip()


def test_readme_matches_committed_results():
    rendered = _load_renderer().build(ART).strip()
    assert _block(README) == rendered, (
        "README results drift: run `python experiments/make_report.py --write` to splice "
        "the renderer's output into the RESULTS block.")


def test_the_renderer_has_no_access_to_the_model_or_the_network():
    """The point of separating renderer from study is that the README cannot invent.

    A future edit that makes ``make_report.py`` import torch (or train, or measure)
    would let a number appear in the README with no artifact behind it.
    """
    source = (ROOT / "experiments" / "make_report.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(torch|relay|numpy)", source,
                         re.MULTILINE), (
        "make_report.py now imports measurement code; every figure it prints has to come "
        "from the committed JSON")


def test_the_renderer_prints_the_arms_the_artifact_measured():
    """The renderer spells out how many harnesses there are, from its own list.

    That is only honest while the list still names the arms the artifact contains, in
    the order the tables print them.
    """
    renderer = _load_renderer()
    assert set(renderer.ORDER) == set(ART["config"]["arms"])
    assert set(renderer.TRAVELS) == set(renderer.ORDER)
    assert set(renderer.SEARCHED) == set(renderer.ORDER)
    assert set(renderer.SWEEP) <= set(renderer.ORDER)
    for arm in renderer.SWEEP:
        assert arm != "single_shot", "single_shot has one budget, so it has no curve"


def test_the_rendered_block_says_nothing_the_artifact_does_not_support():
    """Every three-decimal figure in the block is an artifact number or a difference.

    The renderer is allowed to subtract two measured means -- that is what a gap
    sentence is -- but not to state a rate no measurement produced.
    """
    numbers = {float(m.group(0)) for m in re.finditer(r"(?<![\d.])0\.\d{3}(?![\d])",
                                                      _block(README))}

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, float):
            yield round(node, 3)

    measured = sorted(set(walk(ART)))
    allowed = set(measured)
    allowed |= {round(a - b, 3) for a in measured for b in measured}
    strays = {n for n in numbers if n not in allowed}
    assert not strays, f"the block prints figures the artifact does not hold: {strays}"


def test_no_hand_written_result_survives_outside_the_block():
    """The prose around the block may name the task, not a result.

    ``0.955`` typed into the Design section would be a claim nothing rechecks.
    """
    outside = (README[: README.index("<!-- RESULTS:START -->")]
               + README[README.index("<!-- RESULTS:END -->"):])
    outside = re.sub(r"```.*?```", "", outside, flags=re.DOTALL)  # pinned transcripts
    strays = [m.group(0) for m in re.finditer(r"(?<![\d.])0\.\d{3}(?![\d])", outside)]
    assert not strays, f"hand-written result values outside the results block: {strays}"
