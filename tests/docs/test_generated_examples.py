"""Run examples from current Sphinx output and numerical docstrings."""

import doctest
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

import scicomap as sc
from scicomap import cmath, datasets, scicomap
from scicomap._llm_assets import iter_html_pages, to_markdown


@pytest.mark.parametrize("module", [cmath, datasets, scicomap])
def test_numerical_and_dataset_docstrings(module) -> None:
    try:
        result = doctest.testmod(module)
        assert result.failed == 0
        assert result.attempted > 0
    finally:
        plt.close("all")


@pytest.mark.docs
def test_generated_documentation_examples(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    html_dir = tmp_path / "html"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-n",
            "-W",
            "-b",
            "html",
            str(root / "docs/source"),
            str(html_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    api = (html_dir / "api-reference.html").read_text()
    public_names = [
        name
        for name in sc.__all__
        if name not in {"cmath", "cblind", "datasets"}
    ]
    for name in public_names:
        assert api.count(f'id="scicomap.{name}"') == 1, name
    assert 'id="scicomap.SciCoMap.unif_sym_cmap"' in api
    assert 'id="scicomap.ScicoSequential.assess_cmap"' not in api
    assert 'id="scicomap.cmath._as_color_table"' not in api
    assert "lightness_rounding" in api and "Parameters" in api
    code_pattern = r"^```(?:python|ipython3)\n(.*?)^```$"
    count = 0
    for page in iter_html_pages(html_dir):
        _, markdown = to_markdown(page)
        examples = re.findall(code_pattern, markdown, re.MULTILINE | re.DOTALL)
        if page.name == "tutorial.html":
            notebook = json.loads(
                (root / "docs/source/notebooks/tutorial.ipynb").read_text()
            )
            sources = [
                "\n".join(
                    line.rstrip()
                    for line in "".join(cell["source"]).splitlines()
                )
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            ]
            assert [code.rstrip("\n") for code in examples] == sources
        namespace = {"__name__": "__docs_example__"}
        for index, code in enumerate(examples):
            try:
                exec(
                    compile(code, f"{page.name}:example-{index}", "exec"),
                    namespace,
                )
            finally:
                plt.close("all")
            count += 1
    assert count >= 30, "Missing generated Python examples"
    _, cli_markdown = to_markdown(html_dir / "cli-reference.html")
    assert (
        "| Command | Purpose | Example |\n| --- | --- | --- |" in cli_markdown
    )
    assert "| `scicomap list` |" in cli_markdown
