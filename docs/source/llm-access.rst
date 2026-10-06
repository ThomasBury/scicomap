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

Generate assets from the repository
-----------------------------------

Build or supply the HTML first, then run the documentation maintenance script:

.. code-block:: shell

   uv run --locked python scripts/build_llm_assets.py --html-dir path/to/html

The script calls the packaged generator. Code fences preserve whitespace and
their language, and ordinary documentation tables retain their rows and columns.

Parser assumptions
------------------

- The parser prefers ``<main>`` and supports ``role=\"main\"`` as fallback.
- Sidebar and navigation blocks are excluded by tag and selector rules.
- If your Sphinx theme changes, review parser selectors in
  ``src/scicomap/_llm_assets.py``.

Theme upgrade checklist
-----------------------

Run these checks after changing Sphinx themes or major theme versions:

- ``uv run --locked python -m pytest tests/docs/test_build_llm_assets.py``
- ``just docs check-docs`` (installs documentation extras and runs the strict examples)
