"""Fixed numbers live in constants.py: a numeric module-level constant elsewhere must be allow-listed here."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "retrieval_framework"
SKIPPED_FILES = {"constants.py", "_version.py"}

# "module:NAME" -> reason it may stay outside constants.py
ALLOWED: dict[str, str] = {
    # figure style and display choices
    "report_figures:DPI": "figure resolution",
    "report_figures:TP_PLOT_LOG10_P_TOP": "pressure range of the T-P figure",
    "report_figures:TP_PLOT_N_P": "pressure grid size of the T-P figure",
    "report_figures:TP_PLOT_DRAWS": "number of posterior draws drawn in the T-P figure",
    "report_figures:CORNER_SIGMAS": "contour levels of the corner figure",
    # python floor checked at start-up; mirrors pyproject requires-python
    "validate_env:SUPPORTED_PYTHON": "version tuple",
}


def _is_number(node: ast.AST) -> bool:
    """True for a numeric literal, or a tuple, list, set, dict or arithmetic expression built only from them."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float | complex) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, ast.USub | ast.UAdd) and _is_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_number(node.left) and _is_number(node.right)
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return bool(node.elts) and all(_is_number(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return (
            bool(node.values)
            and all(k is not None and _is_number(k) for k in node.keys)
            and all(_is_number(v) for v in node.values)
        )
    return False


def _module_level_numbers(src: Path) -> dict[str, str]:
    """Map ``module:NAME`` to the source text of every module-level numeric assignment under ``src``."""
    found = {}
    for path in sorted(src.rglob("*.py")):
        if path.name in SKIPPED_FILES:
            continue
        module = ".".join(path.relative_to(src).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if _is_number(value):
                for target in targets:
                    if isinstance(target, ast.Name):
                        found[f"{module}:{target.id}"] = ast.unparse(value)
    return found


def test_no_numeric_module_constants_outside_constants_py():
    found = _module_level_numbers(SRC)
    hidden = [f"  {key} = {value}" for key, value in found.items() if key not in ALLOWED]
    stale = sorted(set(ALLOWED) - set(found))
    assert not hidden, (
        "module-level numbers outside constants.py; move each to constants.py with its unit and reason, "
        "or add it to the allow-list here with a reason:\n" + "\n".join(hidden)
    )
    assert not stale, "allow-list entries that no longer exist; delete them: " + ", ".join(stale)
