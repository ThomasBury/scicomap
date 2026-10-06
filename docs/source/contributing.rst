Contributing
============

Thanks for helping improve scicomap.

Local setup
-----------

Use the ``just`` recipes with ``uv``'s single project ``.venv``. Routine
commands use the committed lockfile without updating it. Run ``uv lock`` only
when changing dependencies and review the lockfile diff.

.. code-block:: shell

   just sync  # locked lint and test extras

Quality checks
--------------

.. code-block:: shell

   just check  # ordinary pytest, Ruff on src/tests/scripts, ty on src/scicomap
   uv run --locked python -m pytest tests/core/test_cmath.py

Tests marked ``docs`` require documentation dependencies and run separately
through ``just check-docs``. Ordinary checks need only lint/test extras.

Build docs and LLM assets
-------------------------

.. code-block:: shell

   just sync-docs  # add docs extras; install the Pandoc binary separately
   just docs
   just check-docs  # strict generated examples and WASM bootstrap tests

Check the installed wheel
-------------------------

After ``just build``, run the smoke check in an isolated environment. It verifies
packaged datasets and canonical commands from a temporary directory outside
the checkout.
Replace the wheel filename below if the package version changes.

.. code-block:: shell

   uv run --isolated --no-project --with ./dist/scicomap-2.0.0-py3-none-any.whl python scripts/smoke_wheel.py

Pull request checklist
----------------------

- Keep changes small and focused.
- Use conventional commit messages.
- Update docs when behavior changes.
- Confirm docs and quality checks pass before requesting review.
