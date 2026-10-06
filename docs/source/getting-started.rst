Getting Started
===============

Start by inspecting the original map. Correct it only when the diagnostics and
its rendering suggest a useful change. Python and CLI examples below select
``hawaii`` from the sequential family and perform the same operations.

Install v2
----------

Python 3.10 or newer is required. Before publication, install this candidate
from the repository root:

.. code-block:: shell

   pip install .

After 2.0.0 is published, install with ``pip install 'scicomap>=2,<3'`` or
``uv add 'scicomap>=2,<3'``. Existing v1 users should read :doc:`migrating-v2`.

Discover and inspect
--------------------

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         import scicomap as sc

         catalog = sc.get_cmap_dict()
         families = list(catalog)
         names = list(catalog["sequential"])
         chart = sc.ScicoSequential("hawaii")
         original = chart.cmap
         diagnostics = sc.diagnose_cmap(original, chart.ctype)
         print(diagnostics["status"])
         figure = chart.assess_cmap(figsize=(14, 6))
         # Save with figure.savefig("hawaii-original.png").

   .. tab:: CLI

      .. code-block:: shell

         scicomap list
         scicomap list sequential
         scicomap check hawaii --type sequential
         scicomap preview hawaii --type sequential --out hawaii-original.png

Diagnostics describe sampled lightness behavior according to the family.
Statuses are heuristics. Inspection leaves the map unchanged. Plotting methods
return Matplotlib Figures without displaying windows; call ``plt.show()``
explicitly or save with ``figure.savefig(...)``.

Correct and compare
-------------------

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         corrected = chart.unif_sym_cmap(lightness_rounding=0, bitonic=False)
         transformed = sc.diagnose_cmap(corrected, chart.ctype)
         assert corrected is chart.cmap
         corrected_figure = chart.assess_cmap(figsize=(14, 6))
         # Save with corrected_figure.savefig("hawaii-corrected.png").

   .. tab:: CLI

      .. code-block:: shell

         scicomap fix hawaii --type sequential --lightness-rounding 0 --no-bitonic --out hawaii-corrected.png
         scicomap report --cmap hawaii --type sequential --fix --lightness-rounding 0 --no-bitonic --out reports/hawaii --json

``lightness_rounding=0`` keeps the lower lightness bound unchanged. Lightness
uniformization and chroma symmetrization are separate operations. The report
stores both original and transformed diagnostics, labels artifacts with the map
used, and includes a reusable corrected color table.

Reuse the exact colors
----------------------

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         import json
         from pathlib import Path
         from tempfile import TemporaryDirectory
         from matplotlib.colors import ListedColormap

         with TemporaryDirectory() as directory:
             path = chart.export_cmap(Path(directory) / "hawaii.json")
             table = json.loads(path.read_text())
         reloaded = ListedColormap(table["rgba"], name=table["name"])
         elevation = sc.datasets.load_hill_topography()
         import matplotlib.pyplot as plt
         fig, ax = plt.subplots()
         ax.imshow(elevation, cmap=reloaded)

   .. tab:: CLI

      The CLI takes an image file. Save the bundled elevation as a grayscale
      image first; use Python to retain the original elevation values.

      .. code-block:: shell

         python -c "import scicomap as sc; import matplotlib.pyplot as plt; plt.imsave('topography.png', sc.datasets.load_hill_topography(), cmap='gray')"
         scicomap fix hawaii --lightness-rounding 0 --no-bitonic --export hawaii.json --json
         scicomap apply hawaii.json --image topography.png --out elevation.png --json

Review simulations and automate
-------------------------------

``chart.colorblind()`` uses the current map, including any correction.
The matching CLI workflow is:

.. code-block:: shell

   scicomap report --cmap hawaii --fix --lightness-rounding 0 --no-bitonic --cvd --out reports/hawaii --json

CVD previews simulate selected color-vision conditions and do not certify
accessibility. Review your actual figure, labels, contrast, and alternate
encodings. See :doc:`user-guide` for normalization and simulation conditions.

Only ``scicomap wizard`` prompts. Every command accepts ``--json``; machine
mode never prompts or opens windows, and rendering requires ``--out``.
See :doc:`cli-reference` for the response envelope and exit codes, or
:doc:`tutorial-marimo` for an interactive walkthrough.
