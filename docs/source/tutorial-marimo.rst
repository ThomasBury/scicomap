Interactive Marimo Tutorial
===========================

Use the Marimo tutorial when you want a guided, reactive experience for
colormap selection, diagnostics, and accessibility checks.

Open the browser tutorial
-------------------------

After docs build, open:

- ``marimo/index.html``

This WASM-powered page runs directly in the browser with no Python backend.

Run the full tutorial locally
-----------------------------

Use the local app when you want richer workflows and larger computations.

.. code-block:: shell

   uv run marimo run docs/marimo/tutorial_app.py

Known WASM constraints
----------------------

- WASM mode supports many, but not all, Python features and packages.
- Browser memory and startup cost can be higher than local mode.
- Use local mode for heavy workflows or if a package limitation appears.

WASM local serving note
-----------------------

When serving exported WASM files locally, serve the full docs root (for example,
``docs/build/html``) rather than only the ``marimo/`` subfolder so absolute
runtime paths resolve consistently.
