LLM Access
==========

scicomap publishes machine-friendly documentation assets alongside HTML pages.

Available assets
----------------

- ``llms.txt`` at the docs root.
- Markdown mirrors for canonical pages under ``/llm/``.

Canonical and preferred formats
-------------------------------

- Canonical user-facing docs are HTML pages.
- Preferred ingestion format for LLM tooling is markdown mirror content.

Stability policy
----------------

- Keep URLs stable across patch releases when possible.
- Add new sections without breaking existing ``llms.txt`` entries.
- Regenerate LLM assets after each docs build.

Generate assets from installed commands
----------------------------------------

Both v1 commands work from an installed wheel. Build or supply the HTML first;
Sphinx and documentation sources are only needed to build the HTML itself.

.. code-block:: shell

   scicomap docs-llm --html-dir path/to/html --json
   scicomap docs llm-assets --html-dir path/to/html --json

The repository script calls the same packaged generator. Code fences preserve
whitespace and their language, and ordinary documentation tables retain their
rows and columns.

Parser assumptions
------------------

- The parser prefers ``<main>`` and supports ``role=\"main\"`` as fallback.
- Sidebar and navigation blocks are excluded by tag and selector rules.
- If your Sphinx theme changes, review parser selectors in
  ``src/scicomap/_llm_assets.py``.

Theme upgrade checklist
-----------------------

Run these checks after changing Sphinx themes or major theme versions:

- ``uv run python -m pytest tests/docs/test_build_llm_assets.py``
- ``uv run sphinx-build -n -b html docs/source docs/build/html``
- ``uv run python scripts/build_llm_assets.py``
