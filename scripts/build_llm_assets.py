#!/usr/bin/env python3
"""Build LLM-friendly assets from generated Sphinx HTML docs.

The parser prefers semantic content containers (``<main>`` and ``role="main"``)
and falls back to common documentation wrappers used by Sphinx themes.
If theme markup changes, update ``CONTENT_CONTAINER_CLASSES`` and skip selectors.
"""

from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator, TypedDict


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
    "autoapi",
}
CONTENT_CONTAINER_CLASSES = {
    "bd-main",
    "document",
    "documentwrapper",
    "main",
    "wy-nav-content",
    "wy-nav-content-wrap",
}
THEME_TITLE_SUFFIX = " | scicomap documentation"
PRIORITY_PAGES = [
    "index.html",
    "getting-started.html",
    "user-guide.html",
    "api-reference.html",
    "faq.html",
    "troubleshooting.html",
    "notebooks/tutorial.html",
]


class DocEntry(TypedDict):
    """Metadata for one canonical documentation page.

    Attributes
    ----------
    title : str
        Human-readable page title extracted from HTML.
    html_rel : str
        Relative path to the HTML page from the docs root.
    md_rel : str
        Relative path to the generated markdown page from the docs root.
    """

    title: str
    html_rel: str
    md_rel: str


class HtmlToMarkdownParser(HTMLParser):
    """Extract readable markdown-like text from an HTML document.

    Notes
    -----
    This parser relies on common Sphinx theme structure. Review content and
    sidebar selectors when changing or upgrading the docs theme.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_tag_stack: list[str] = []
        self._skip_depth = 0
        self._content_tag_stack: list[str] = []
        self._heading_level = 0
        self._in_code = False
        self._in_list_item = False
        self._in_title = False
        self._text_chunks: list[str] = []
        self._current_link: str | None = None
        self.title = ""
        self.blocks: list[str] = []

    def _enter_skip(self, tag: str) -> None:
        """Start ignoring content until matching closing tag.

        Parameters
        ----------
        tag : str
            Name of the tag that started the skipped region.
        """
        self._skip_tag_stack.append(tag)
        self._skip_depth = len(self._skip_tag_stack)

    def _exit_skip(self, tag: str) -> None:
        """Stop ignoring content when a skipped region closes.

        Parameters
        ----------
        tag : str
            Name of the closing tag.
        """
        if not self._skip_tag_stack:
            return

        for index in range(len(self._skip_tag_stack) - 1, -1, -1):
            if self._skip_tag_stack[index] == tag:
                del self._skip_tag_stack[index:]
                break

        self._skip_depth = len(self._skip_tag_stack)

    def _in_content(self) -> bool:
        """Return whether parser is inside a content container."""
        return bool(self._content_tag_stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle opening HTML tags during parsing.

        Parameters
        ----------
        tag : str
            Name of the opening tag.
        attrs : list[tuple[str, str | None]]
            Parsed attributes for the current tag.
        """
        attrs_dict = dict(attrs)
        classes = set((attrs_dict.get("class") or "").split())
        role = attrs_dict.get("role")

        if self._skip_tag_stack:
            if tag in {"script", "style", "noscript"}:
                self._enter_skip(tag)
            elif tag in {"nav", "header", "footer", "aside"}:
                self._enter_skip(tag)
            elif "wy-nav-side" in classes:
                self._enter_skip(tag)
            return

        if tag in {"script", "style", "noscript"}:
            self._enter_skip(tag)
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "main" or role == "main" or classes.intersection(CONTENT_CONTAINER_CLASSES):
            self._content_tag_stack.append(tag)

        if "wy-nav-side" in classes:
            self._enter_skip(tag)
            return

        if tag in {"nav", "header", "footer", "aside"}:
            self._enter_skip(tag)
            return

        if not self._in_content():
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
        """Handle closing HTML tags and emit markdown blocks.

        Parameters
        ----------
        tag : str
            Name of the closing tag.
        """
        if tag == "title":
            self._in_title = False
            return

        if self._skip_tag_stack:
            self._exit_skip(tag)
            return

        if self._content_tag_stack and self._content_tag_stack[-1] == tag:
            self._content_tag_stack.pop()
            if not self._in_content():
                return

        if not self._in_content():
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
        """Handle text data from HTML tags.

        Parameters
        ----------
        data : str
            Raw text payload from the HTML parser.
        """
        if self._skip_tag_stack:
            return

        cleaned = data.strip()
        if not cleaned:
            return

        if self._in_title:
            self.title = cleaned
            return

        if not self._in_content():
            return

        if self._current_link:
            cleaned = f"{cleaned} ({self._current_link})"
        elif self._in_code:
            cleaned = f"`{cleaned}`"

        self._text_chunks.append(cleaned)


def iter_html_pages(html_dir: Path) -> Iterator[Path]:
    """Yield canonical HTML pages that should be mirrored.

    Parameters
    ----------
    html_dir : Path
        Root directory containing generated Sphinx HTML output.

    Yields
    ------
    Path
        Paths to HTML pages that are included in the markdown mirror.
    """
    for html_file in sorted(html_dir.rglob("*.html")):
        rel = html_file.relative_to(html_dir)
        if html_file.name in EXCLUDED_FILENAMES:
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
            continue
        yield html_file


def to_markdown(html_path: Path) -> tuple[str, str]:
    """Convert one HTML page into markdown content.

    Parameters
    ----------
    html_path : Path
        Path to a generated Sphinx HTML page.

    Returns
    -------
    tuple[str, str]
        A tuple with the resolved page title and markdown payload.
    """
    parser = HtmlToMarkdownParser()
    parser.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
    title = parser.title or html_path.stem.replace("_", " ").title()
    if title.endswith(THEME_TITLE_SUFFIX):
        title = title.removesuffix(THEME_TITLE_SUFFIX)

    blocks = list(parser.blocks)
    if blocks and blocks[0].startswith("# "):
        first_h1 = blocks[0][2:].strip()
        first_h1 = first_h1.split(" (#", maxsplit=1)[0].rstrip(" #").strip()
        if first_h1 == title:
            content = blocks
        else:
            content = [f"# {title}", *blocks]
    else:
        content = [f"# {title}", *blocks]

    markdown = "\n\n".join(content).strip() + "\n"
    return title, markdown


def build_markdown_mirror(html_dir: Path, markdown_dir: Path) -> list[DocEntry]:
    """Generate markdown mirrors for all selected HTML pages.

    Parameters
    ----------
    html_dir : Path
        Root directory containing generated Sphinx HTML output.
    markdown_dir : Path
        Destination directory where markdown mirrors are written.

    Returns
    -------
    list[DocEntry]
        Metadata for all generated markdown pages.
    """
    generated: list[DocEntry] = []
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


def write_llms_txt(html_dir: Path, base_url: str, docs: list[DocEntry]) -> None:
    """Write an ``llms.txt`` manifest for generated docs.

    Parameters
    ----------
    html_dir : Path
        Root directory containing generated Sphinx HTML output.
    base_url : str
        Canonical public base URL for the documentation site.
    docs : list[DocEntry]
        Metadata describing each generated markdown mirror.
    """
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    doc_map = {doc["html_rel"]: doc for doc in docs}
    ordered: list[DocEntry] = []

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
        "## Start Here",
    ]

    for page in PRIORITY_PAGES:
        doc = next((item for item in ordered if item["html_rel"] == page), None)
        if doc is None:
            continue
        lines.append(f"- {doc['title']}: {base_url.rstrip('/')}/{doc['md_rel']}")

    lines.extend(
        [
            "",
        "## Canonical Documents",
        ]
    )

    for doc in ordered:
        html_url = f"{base_url.rstrip('/')}/{doc['html_rel']}"
        md_url = f"{base_url.rstrip('/')}/{doc['md_rel']}"
        lines.append(f"- {doc['title']}")
        lines.append(f"  - html: {html_url}")
        lines.append(f"  - markdown: {md_url}")

    (html_dir / "llms.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the asset build script.

    Returns
    -------
    argparse.Namespace
        Parsed options for HTML source directory and base URL.
    """
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
    """Build markdown mirrors and an ``llms.txt`` manifest.

    Returns
    -------
    int
        Process exit code. Returns ``0`` on success.

    Raises
    ------
    FileNotFoundError
        If the input HTML directory does not exist.
    RuntimeError
        If no eligible HTML pages are found for conversion.
    """
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
