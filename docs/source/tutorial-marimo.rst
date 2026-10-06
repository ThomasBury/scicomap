Interactive Marimo Tutorial
===========================

Use the Marimo tutorial when you want a guided, reactive experience for
colormap selection, lightness diagnostics, and color-vision simulations.

The tutorials and CLI share family-aware diagnostic heuristics. Their statuses
and CVD simulations do not certify accessibility. In both tutorials,
diagnostics, assessment, simulations, and sample images use the same selected
map, including the correction when enabled.

Prefer a linear, narrative walkthrough? See :doc:`notebooks/tutorial`.

Two app variants are maintained:

- ``docs/marimo/tutorial_app.py``: full local workflow with richer controls.
- ``docs/marimo/tutorial_app_lite.py``: browser WASM workflow for docs hosting.

Open the browser tutorial
-------------------------

After docs build, open:

- ``marimo/index.html``
- `https://thomasbury.github.io/scicomap/marimo/index.html <https://thomasbury.github.io/scicomap/marimo/index.html>`_

This WASM-powered page runs directly in the browser with no Python backend.

Run the full tutorial locally
-----------------------------

Use the local app when you want richer workflows and larger computations.

.. code-block:: shell

   just sync-docs
   uv run --locked marimo run docs/marimo/tutorial_app.py

Known WASM constraints
----------------------

- WASM mode supports many, but not all, Python features and packages.
- Browser memory and startup cost can be higher than local mode.
- Use local mode for heavy workflows or if a package limitation appears.
- ``just marimo`` builds this checkout as ``scicomap-2.0.0-py3-none-any.whl``
  in ``marimo/public``. The browser installs that candidate and its dependencies
  before importing the public APIs; the heading shows the installed version.
  No published v1 installation or copied diagnostic module is used.

WASM local serving note
-----------------------

When serving exported WASM files locally, serve the full docs root (for example,
``docs/build/html``) rather than only the ``marimo/`` subfolder so absolute
runtime paths resolve consistently.

Next steps from this tutorial
-----------------------------

- Quick install and first commands: :doc:`getting-started`
- Practical decision workflow: :doc:`user-guide`
- Deeper narrative tutorial: :doc:`notebooks/tutorial`

Verify the candidate in a browser
---------------------------------

.. code-block:: shell

   just docs marimo validate-doc-artifacts
   uv run --locked python -m http.server 8000 --bind 127.0.0.1 --directory docs/build/html

Open ``http://127.0.0.1:8000/marimo/index.html``. Wait for dependency
installation, confirm the heading shows ``scicomap 2.0.0``, and switch family
and map. Enable ``Apply fix`` and check that diagnostics, assessment, and CVD
panels update together. The CLI tab lists the matching stage choices. Test
sequential and diverging maps as well as an unordered qualitative palette.
The first load needs Internet access for Pyodide and dependencies.
