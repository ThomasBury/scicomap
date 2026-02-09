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
