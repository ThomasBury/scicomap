"""Check the shared diagnostic module's browser bootstrap and import path."""

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from scicomap._diagnostics import _diagnose_cmap
from scicomap.scicomap import SciCoMap


def test_wasm_diagnostics_install_before_import(tmp_path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/scicomap/_diagnostics.py").read_text()
    tutorial = ast.parse(
        (root / "docs/marimo/tutorial_app_lite.py").read_text()
    )
    initialize = next(
        node
        for node in tutorial.body
        if isinstance(node, ast.AsyncFunctionDef)
    )
    initialize.decorator_list = []
    installed = []

    async def install(requirement):
        installed.append(requirement)

    async def fetch(url):
        assert installed == ["scicomap>=1.1.0"]
        assert (
            url
            == "https://example.test/scicomap/marimo/public/_diagnostics.py"
        )

        async def string():
            return source

        return SimpleNamespace(ok=True, string=string)

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
    monkeypatch.setitem(
        sys.modules, "pyodide.http", SimpleNamespace(pyfetch=fetch)
    )
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(
        compile(
            ast.Module(body=[initialize], type_ignores=[]), "bootstrap", "exec"
        ),
        namespace,
    )
    assert asyncio.run(namespace["_"]()) == (True,)
    module_path = tmp_path / "_scicomap_diagnostics.py"
    assert module_path.read_text() == source
    spec = importlib.util.spec_from_file_location(
        "_scicomap_diagnostics", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cmap = SciCoMap(ctype="diverging", cmap="wildfire").get_mpl_color_map()
    assert module._diagnose_cmap(cmap, "diverging") == _diagnose_cmap(
        cmap, "diverging"
    )
