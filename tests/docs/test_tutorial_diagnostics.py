"""Verify candidate-wheel bootstrap and shared tutorial map selection."""

import ast
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scicomap as sc


ROOT = Path(__file__).resolve().parents[2]


def cells(filename):
    """Read actual tutorial cells without starting the app."""
    return [
        node
        for node in ast.parse(
            (ROOT / "docs/marimo" / filename).read_text()
        ).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def run_cell(node, **kwargs):
    """Execute one actual cell with explicit dependencies."""
    node.decorator_list = []
    namespace = {}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "tutorial", "exec"),
        namespace,
    )
    return namespace["_"](**kwargs)


@pytest.mark.docs
def test_wasm_candidate_install_before_import(monkeypatch) -> None:
    tutorial = cells("tutorial_app_lite.py")
    installed = []

    async def install(requirement):
        installed.append(requirement)

    monkeypatch.setitem(
        sys.modules, "micropip", SimpleNamespace(install=install)
    )
    monkeypatch.setitem(
        sys.modules,
        "js",
        SimpleNamespace(
            location="https://example.test/scicomap/marimo/assets/worker.js"
        ),
    )
    monkeypatch.setattr("marimo._runtime.runtime.is_pyodide", lambda: True)
    assert asyncio.run(run_cell(tutorial[0])) == (True,)
    assert installed == [
        "https://example.test/scicomap/marimo/public/scicomap-2.0.0-py3-none-any.whl"
    ]
    imported = run_cell(tutorial[1], wasm_deps_ready=True)
    assert imported[0] == "2.0.0"
    assert imported[4] is sc.diagnose_cmap
    monkeypatch.setattr(sc, "__version__", "1.1.1")
    with pytest.raises(RuntimeError, match="bundled v2 candidate"):
        run_cell(tutorial[1], wasm_deps_ready=True)


@pytest.mark.parametrize(
    "filename", ["tutorial_app.py", "tutorial_app_lite.py"]
)
@pytest.mark.parametrize("fix", [False, True])
def test_tutorial_uses_selected_map_once(filename, fix, monkeypatch) -> None:
    tutorial = cells(filename)
    select = next(
        node
        for node in tutorial
        if {arg.arg for arg in node.args.args}
        == {
            "SciCoMap",
            "bitonic",
            "cmap",
            "ctype",
            "diffuse",
            "fix",
            "lightness_rounding",
        }
    )
    transforms = []
    actual = sc.SciCoMap.unif_sym_cmap

    def transform(self, **kwargs):
        transforms.append(kwargs)
        return actual(self, **kwargs)

    monkeypatch.setattr(sc.SciCoMap, "unif_sym_cmap", transform)
    value = lambda v: SimpleNamespace(value=v)
    chart, selected = run_cell(
        select,
        SciCoMap=sc.SciCoMap,
        bitonic=value(False),
        cmap=value("hawaii"),
        ctype=value("sequential"),
        diffuse=value(True),
        fix=value(fix),
        lightness_rounding=value(0),
    )
    assert selected is chart.cmap
    controls = next(
        node
        for node in tutorial
        if "controls"
        in {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        and "cmap" in {arg.arg for arg in node.args.args}
    )
    dropdown = value("hawaii")
    dependencies = {arg.arg: value(False) for arg in controls.args.args}
    dependencies.update(
        cmap=dropdown,
        mo=SimpleNamespace(
            md=lambda text: text, vstack=lambda items, **kw: items
        ),
    )
    assert dropdown in run_cell(controls, **dependencies)[0]
    assert len(transforms) == int(fix)
    diagnose = next(
        node
        for node in tutorial
        if {arg.arg for arg in node.args.args}
        == {
            "diagnose_cmap",
            "ctype",
            "selected_map",
        }
    )
    assert run_cell(
        diagnose,
        diagnose_cmap=sc.diagnose_cmap,
        ctype=value("sequential"),
        selected_map=selected,
    ) == (sc.diagnose_cmap(selected, "sequential"),)
    simulate = next(
        node
        for node in tutorial
        if {arg.arg for arg in node.args.args}
        == {
            "ctype",
            "n_colors",
            "plot_colorblind_vision",
            "selected_map",
        }
    )

    def plot(**kwargs):
        assert kwargs["cmap_list"] == [selected]
        assert not kwargs["uniformize"] and not kwargs["symmetrize"]
        return "figure"

    assert run_cell(
        simulate,
        ctype=value("sequential"),
        n_colors=value(128),
        plot_colorblind_vision=plot,
        selected_map=selected,
    ) == ("figure",)
