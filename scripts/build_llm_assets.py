#!/usr/bin/env python3
"""Build LLM-friendly assets from generated Sphinx HTML docs."""

from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


EXCLUDED_FILENAMES = {
    "genindex.html",
    "py-modindex.html",
    "search.html",
}
EXCLUDED_DIR_NAMES = {
    "_images",
    "_modules",
    "_sources",
    "_static",
}
PRIORITY_PAGES = [
    "index.html",
    "Introduction.html",
    "notebooks/tutorial.html",
    "autoapi/index.html",
]


class HtmlToMarkdownParser(HTMLParser):
    """Extract readable markdown-like text from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_main = False
        self._heading_level = 0
        self._in_code = False
        self._in_list_item = False
        self._in_title = False
        self._text_chunks: list[str] = []
        self._current_link: str | None = None
        self.title = ""
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())

        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "main":
            self._in_main = True

        if "wy-nav-side" in classes:
            self._skip_depth += 1
            return

        if tag in {"nav", "header", "footer", "aside"}:
            self._skip_depth += 1
            return

        if not self._in_main:
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._text_chunks = []
        elif tag == "li":
            self._in_list_item = True
            self._text_chunks = []
        elif tag in {"p", "pre"}:
            self._text_chunks = []
            self._in_code = tag == "pre"
        elif tag == "code":
            self._in_code = True
        elif tag == "a":
            self._current_link = attrs_dict.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return

        if tag in {"script", "style", "noscript", "nav", "header", "footer", "aside"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if self._skip_depth > 0:
            return

        if tag == "main":
            self._in_main = False
            return

        if not self._in_main:
            return

        text = " ".join(chunk.strip() for chunk in self._text_chunks if chunk.strip())
        if not text:
            if tag in {"p", "pre", "li", "code", "h1", "h2", "h3", "h4", "h5", "h6"}:
                self._text_chunks = []
            if tag in {"pre", "code"}:
                self._in_code = False
            if tag == "li":
                self._in_list_item = False
            if tag == "a":
                self._current_link = None
            return

        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.blocks.append(f"{'#' * self._heading_level} {text}")
            self._heading_level = 0
            self._text_chunks = []
        elif tag == "li":
            self.blocks.append(f"- {text}")
            self._in_list_item = False
            self._text_chunks = []
        elif tag == "pre":
            self.blocks.append(f"```\n{text}\n```")
            self._in_code = False
            self._text_chunks = []
        elif tag == "p":
            self.blocks.append(text)
            self._text_chunks = []
        elif tag == "code":
            self._in_code = False
        elif tag == "a":
            self._current_link = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        cleaned = data.strip()
        if not cleaned:
            return

        if self._in_title:
            self.title = cleaned
            return

        if not self._in_main:
            return

        if self._current_link:
            cleaned = f"{cleaned} ({self._current_link})"
        elif self._in_code:
            cleaned = f"`{cleaned}`"

        self._text_chunks.append(cleaned)


def iter_html_pages(html_dir: Path) -> Iterable[Path]:
    for html_file in sorted(html_dir.rglob("*.html")):
        rel = html_file.relative_to(html_dir)
        if html_file.name in EXCLUDED_FILENAMES:
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        yield html_file


def to_markdown(html_path: Path) -> tuple[str, str]:
    parser = HtmlToMarkdownParser()
    parser.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
    title = parser.title or html_path.stem.replace("_", " ").title()

    content = [f"# {title}"]
    for block in parser.blocks:
        content.append(block)

    markdown = "\n\n".join(content).strip() + "\n"
    return title, markdown


def build_markdown_mirror(html_dir: Path, markdown_dir: Path) -> list[dict[str, str]]:
    generated: list[dict[str, str]] = []
    markdown_dir.mkdir(parents=True, exist_ok=True)

    for html_path in iter_html_pages(html_dir):
        rel_html = html_path.relative_to(html_dir)
        md_path = markdown_dir / rel_html.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        title, markdown = to_markdown(html_path)
        md_path.write_text(markdown, encoding="utf-8")

        generated.append(
            {
                "title": title,
                "html_rel": rel_html.as_posix(),
                "md_rel": md_path.relative_to(html_dir).as_posix(),
            }
        )

    return generated


def write_llms_txt(html_dir: Path, base_url: str, docs: list[dict[str, str]]) -> None:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    doc_map = {doc["html_rel"]: doc for doc in docs}
    ordered: list[dict[str, str]] = []

    for page in PRIORITY_PAGES:
        if page in doc_map:
            ordered.append(doc_map.pop(page))

    ordered.extend(sorted(doc_map.values(), key=lambda item: item["html_rel"]))

    lines = [
        "# llms.txt",
        "project: scicomap",
        f"base_url: {base_url.rstrip('/')}/",
        "description: Scientific colormap tooling documentation.",
        "preferred_format: markdown",
        f"generated_utc: {now}",
        "",
        "## Canonical Documents",
    ]

    for doc in ordered:
        html_url = f"{base_url.rstrip('/')}/{doc['html_rel']}"
        md_url = f"{base_url.rstrip('/')}/{doc['md_rel']}"
        lines.append(f"- {doc['title']}")
        lines.append(f"  - html: {html_url}")
        lines.append(f"  - markdown: {md_url}")

    (html_dir / "llms.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html-dir",
        default="docs/build/html",
        type=Path,
        help="Directory containing built Sphinx HTML docs.",
    )
    parser.add_argument(
        "--base-url",
        default="https://thomasbury.github.io/scicomap",
        help="Canonical documentation base URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    html_dir = args.html_dir.resolve()
    markdown_dir = html_dir / "llm"

    if not html_dir.exists():
        raise FileNotFoundError(f"Missing HTML directory: {html_dir}")

    docs = build_markdown_mirror(html_dir=html_dir, markdown_dir=markdown_dir)
    if not docs:
        raise RuntimeError("No HTML pages were converted to markdown.")

    write_llms_txt(html_dir=html_dir, base_url=args.base_url, docs=docs)
    print(f"Generated {len(docs)} markdown mirrors in {markdown_dir}")
    print(f"Generated llms.txt at {html_dir / 'llms.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
