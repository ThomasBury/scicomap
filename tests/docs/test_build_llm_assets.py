"""Regression tests for LLM docs asset generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "build_llm_assets.py"
    spec = importlib.util.spec_from_file_location("build_llm_assets", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sidebar_skip_is_balanced_for_div_wrappers() -> None:
    module = _load_module()
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
    assert all("Ignore this sidebar text." not in block for block in parser.blocks)
    assert parser._skip_depth == 0


def test_nested_skipped_regions_resume_after_close() -> None:
    module = _load_module()
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
    assert all("Skip A" not in block and "Skip B" not in block for block in parser.blocks)
    assert parser._skip_depth == 0


def test_to_markdown_does_not_duplicate_h1(tmp_path: Path) -> None:
    module = _load_module()
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
    h1_lines = [line for line in markdown.splitlines() if line.startswith("# ")]

    assert title == "Getting Started"
    assert h1_lines == ["# Getting Started # (#getting-started)"]
    assert markdown.endswith("\n")


def test_parser_supports_role_main_without_main_tag() -> None:
    module = _load_module()
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
