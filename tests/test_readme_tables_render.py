"""Every pipe table in README.md must be structurally valid GitHub markdown.

The byte-comparison test pins the results block to the committed artifact, so it
cannot notice a renderer that emits a separator row with its cells glued together --
GitHub then silently shows the whole table as literal text. This checks the *shape* of
each table instead: consistent column counts, a separator row of ``-``/``:`` cells with
one cell per column, and a results block whose tables carry the columns the renderer
claims to print.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _cells(line: str) -> list[str] | None:
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return None
    return [c.strip() for c in s[1:-1].split("|")]


def _tables(text: str):
    block: list[tuple[int, list[str]]] = []
    for n, line in enumerate(text.splitlines(), 1):
        cells = _cells(line)
        if cells is None:
            if block:
                yield block
            block = []
        else:
            block.append((n, cells))
    if block:
        yield block


def _renderer():
    path = ROOT / "experiments" / "make_report.py"
    spec = importlib.util.spec_from_file_location("make_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_contains_tables():
    assert list(_tables(README.read_text(encoding="utf-8"))), \
        "README has no pipe tables; this test would pass vacuously"


def test_every_table_is_structurally_valid():
    problems = []
    for block in _tables(README.read_text(encoding="utf-8")):
        header_line, header = block[0]
        widths = {len(cells) for _, cells in block}
        if len(widths) > 1:
            problems.append(f"line {header_line}: ragged column counts {sorted(widths)}")
        if len(block) < 2:
            problems.append(f"line {header_line}: table has no separator row")
            continue
        sep_line, sep = block[1]
        if not all(re.fullmatch(r":?-+:?", cell) for cell in sep):
            problems.append(f"line {sep_line}: separator row is not all dashes: {sep}")
        elif len(sep) != len(header):
            problems.append(f"line {sep_line}: separator has {len(sep)} cells, header "
                            f"{header_line} has {len(header)}")
    assert not problems, "README tables GitHub cannot render:\n" + "\n".join(problems)


def test_every_table_row_repeats_the_header_width():
    for block in _tables(README.read_text(encoding="utf-8")):
        _, header = block[0]
        for line, cells in block[2:]:
            assert len(cells) == len(header), f"line {line}"


def test_the_results_tables_have_one_row_per_harness():
    """A row silently dropped by the renderer would leave a shorter story, not an error."""
    data = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    block = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->",
                      README.read_text(encoding="utf-8"), re.DOTALL).group(1)
    tables = list(_tables(block))
    by_arm = [t for t in tables if t[0][1][0] == "harness"]
    assert len(by_arm) == 2, "the arm table and the per-depth table"
    for table in by_arm:
        assert [row[1][0] for row in table[2:]] == [f"`{a}`"
                                                    for a in _renderer().ORDER]
        assert len(table) - 2 == len(data["config"]["arms"])


def test_the_budget_grid_rows_are_the_budgets_the_study_swept():
    data = json.loads((ROOT / "results" / "frontier.json").read_text(encoding="utf-8"))
    block = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->",
                      README.read_text(encoding="utf-8"), re.DOTALL).group(1)
    grids = [t for t in _tables(block)
             if t[0][1][0].startswith("notebook budget")]
    assert len(grids) == 2, "answer accuracy and lost-in-relay, same axes"
    budgets = sorted({b for arm in _renderer().SWEEP
                      for b in data["config"]["budgets"][arm]})
    for grid in grids:
        assert [row[1][0] for row in grid[2:]] == [str(b) for b in budgets]
        assert grid[0][1][1:] == [f"`{a}`" for a in _renderer().SWEEP]
        assert len(grid[0][1]) == len(_renderer().SWEEP) + 1


def test_no_cell_is_glued_or_empty_in_the_rendered_block():
    block = re.search(r"<!-- RESULTS:START -->\n(.*?)\n<!-- RESULTS:END -->",
                      README.read_text(encoding="utf-8"), re.DOTALL).group(1)
    for table in _tables(block):
        for line, cells in table:
            assert all(c for c in cells), f"line {line}: empty cell in {cells}"
