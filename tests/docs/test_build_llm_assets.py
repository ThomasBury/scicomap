"""Regression tests for LLM docs asset generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scicomap import _llm_assets as module
from scicomap.cli import app


def test_sidebar_skip_is_balanced_for_div_wrappers() -> None:
    parser = module.HtmlToMarkdownParser()
    parser.feed(
        """
        <html>
          <head><title>Example | scicomap documentation</title></head>
          <body>
            <main>
              <div class="wy-nav-side"><p>Ignore this sidebar text.</p></div>
              <h1>Example # (#example)</h1>
              <p>Keep this paragraph.</p>
            </main>
          </body>
        </html>
        """
    )

    assert parser.blocks[0].startswith("# Example")
    assert "Keep this paragraph." in parser.blocks
    assert all(
        "Ignore this sidebar text." not in block for block in parser.blocks
    )
    assert parser._skip_depth == 0


def test_nested_skipped_regions_resume_after_close() -> None:
    parser = module.HtmlToMarkdownParser()
    parser.feed(
        """
        <main>
          <div class="wy-nav-side">
            <nav><p>Skip A</p></nav>
            <aside><p>Skip B</p></aside>
          </div>
          <p>Keep after nested skip.</p>
        </main>
        """
    )

    assert "Keep after nested skip." in parser.blocks
    assert all(
        "Skip A" not in block and "Skip B" not in block
        for block in parser.blocks
    )
    assert parser._skip_depth == 0


def test_to_markdown_does_not_duplicate_h1(tmp_path: Path) -> None:
    html = tmp_path / "getting-started.html"
    html.write_text(
        """
        <html>
          <head><title>Getting Started | scicomap documentation</title></head>
          <body>
            <main>
              <h1>Getting Started # (#getting-started)</h1>
              <p>One paragraph.</p>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    title, markdown = module.to_markdown(html)
    h1_lines = [
        line for line in markdown.splitlines() if line.startswith("# ")
    ]

    assert title == "Getting Started"
    assert h1_lines == ["# Getting Started # (#getting-started)"]
    assert markdown.endswith("\n")


def test_parser_supports_role_main_without_main_tag() -> None:
    parser = module.HtmlToMarkdownParser()
    parser.feed(
        """
        <html>
          <head><title>Fallback</title></head>
          <body>
            <div role="main">
              <h1>Fallback</h1>
              <p>Role main content works.</p>
            </div>
          </body>
        </html>
        """
    )

    assert "# Fallback" in parser.blocks
    assert "Role main content works." in parser.blocks


def test_tutorial_notebook_image_references_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    notebook = root / "docs" / "source" / "notebooks" / "tutorial.ipynb"
    doc_source = root / "docs" / "source"

    payload = json.loads(notebook.read_text(encoding="utf-8"))
    refs: list[str] = []

    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", []):
            marker = '<img src="'
            if marker not in line:
                continue
            path = line.split(marker, 1)[1].split('"', 1)[0]
            if "pics/" in path:
                refs.append(path)

    assert refs, "No tutorial image references found."
    for ref in refs:
        target = (doc_source / "notebooks" / ref).resolve()
        assert target.exists(), f"Missing tutorial image reference: {ref}"


def test_highlighted_code_preserves_whitespace_and_inline_tokens() -> None:
    parser = module.HtmlToMarkdownParser()
    parser.feed(
        '<main><div><div class="highlight-python"><div class="highlight">'
        '<pre><span></span><span class="k">def</span> <span>demo</span>():\n'
        "\tvalue = &quot;a  b&quot;\n\n"
        "\treturn value &lt; &quot;c&quot;\n</pre></div></div></div>"
        "<p>Use <code><span>demo</span>()</code> here.</p></main>"
    )
    assert parser.blocks == [
        '```python\ndef demo():\n\tvalue = "a  b"\n\n'
        '\treturn value < "c"\n```',
        "Use `demo()` here.",
    ]
    compile(
        parser.blocks[0].split("\n", 1)[1].rsplit("\n", 1)[0],
        "example",
        "exec",
    )


def test_table_keeps_cells_rows_and_inline_code() -> None:
    parser = module.HtmlToMarkdownParser()
    parser.feed(
        "<main><table><thead><tr><th><p>Command</p></th><th>Purpose</th>"
        "</tr></thead><tbody><tr><td><p><code>scicomap list</code></p></td>"
        "<td><p>List families</p><p>or names | aliases.</p></td></tr>"
        "</tbody></table><p>After table.</p></main>"
    )
    assert parser.blocks == [
        "| Command | Purpose |\n| --- | --- |\n"
        "| `scicomap list` | List families or names \\| aliases. |",
        "After table.",
    ]


@pytest.mark.parametrize("command", [["docs-llm"], ["docs", "llm-assets"]])
def test_docs_commands_generate_assets(command, tmp_path: Path) -> None:
    html = tmp_path / "index.html"
    html.write_text("<main><h1>Example</h1><p>Content.</p></main>")
    result = CliRunner().invoke(
        app, [*command, "--html-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["data"]["generated_pages"] == 1
    assert (
        tmp_path / "llm/index.md"
    ).read_text() == "# Index\n\n# Example\n\nContent.\n"
    assert (tmp_path / "llms.txt").exists()


def test_docs_command_missing_directory_does_not_create_it(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    result = CliRunner().invoke(
        app, ["docs-llm", "--html-dir", str(missing), "--json"]
    )
    assert result.exit_code == 2
    assert json.loads(result.stdout)["ok"] is False
    assert not missing.exists()
