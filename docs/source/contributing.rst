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
   uv run ruff check src tests
   uv run ruff format --check src tests

Build docs and LLM assets
-------------------------

.. code-block:: shell

   uv run sphinx-build -n -W -b html docs/source docs/build/html
   uv run python scripts/build_llm_assets.py

Check the installed wheel
-------------------------

After ``just build``, run the smoke check in an isolated environment. It verifies
packaged datasets and v1 commands from a temporary directory outside the checkout.
Replace the wheel filename below if the package version changes.

.. code-block:: shell

   uv run --isolated --no-project --with ./dist/scicomap-1.1.1-py3-none-any.whl python scripts/smoke_wheel.py

Pull request checklist
----------------------

- Keep changes small and focused.
- Use conventional commit messages.
- Update docs when behavior changes.
- Confirm docs and quality checks pass before requesting review.
