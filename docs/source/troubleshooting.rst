Troubleshooting
===============

Installation fails
------------------

- Confirm you use a supported Python version (see ``pyproject.toml``).
- Upgrade pip and retry: ``python -m pip install --upgrade pip``.

Docs build fails with ``PandocMissing``
---------------------------------------

The notebook build uses ``nbsphinx`` and requires a pandoc binary.

- Local: install pandoc from https://pandoc.org/installing.html
- CI: use a setup step such as ``r-lib/actions/setup-pandoc``

Docs build reports missing images
---------------------------------

Check image paths in notebooks and RST files. Paths are resolved from
``docs/source`` during Sphinx builds.

API reference page is empty
---------------------------

- Ensure the package imports in the docs environment.
- Confirm Sphinx can resolve ``src`` in ``docs/source/conf.py``.
- Rebuild with verbosity: ``sphinx-build -n -b html docs/source docs/build/html``
