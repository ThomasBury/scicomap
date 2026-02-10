Contributing
============

Thanks for helping improve scicomap.

Local setup
-----------

Use ``uv`` for reproducible local development.

.. code-block:: shell

   uv sync --extra lint --extra test --extra docs

Quality checks
--------------

.. code-block:: shell

   uv run python -m pytest
   uv run python -m flake8 src
   uv run python -m black --check src

Build docs and LLM assets
-------------------------

.. code-block:: shell

   uv run sphinx-build -n -b html docs/source docs/build/html
   uv run python scripts/build_llm_assets.py

Pull request checklist
----------------------

- Keep changes small and focused.
- Use conventional commit messages.
- Update docs when behavior changes.
- Confirm docs and quality checks pass before requesting review.
