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

Parser assumptions
------------------

- The parser prefers ``<main>`` and supports ``role=\"main\"`` as fallback.
- Sidebar and navigation blocks are excluded by tag and selector rules.
- If your Sphinx theme changes, review parser selectors in
  ``scripts/build_llm_assets.py``.

Theme upgrade checklist
-----------------------

Run these checks after changing Sphinx themes or major theme versions:

- ``uv run python -m pytest tests/docs/test_build_llm_assets.py``
- ``uv run sphinx-build -n -b html docs/source docs/build/html``
- ``uv run python scripts/build_llm_assets.py``
