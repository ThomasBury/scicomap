CLI Reference
=============

Use these commands for the most common scicomap workflows.

Quick command map
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 44 32

   * - Command
     - Purpose
     - Example
   * - ``scicomap list``
     - List colormap families or family-specific names.
     - ``scicomap list sequential``
   * - ``scicomap check``
     - Diagnose one colormap and return status/reasons.
     - ``scicomap check hawaii --type sequential``
   * - ``scicomap preview``
     - Render a visual assessment panel.
     - ``scicomap preview hawaii --type sequential --out hawaii.png``
   * - ``scicomap compare``
     - Compare multiple colormaps on one image.
     - ``scicomap compare hawaii viridis --type sequential --image scan --out compare.png``
   * - ``scicomap fix``
     - Uniformize/symmetrize and preview a colormap.
     - ``scicomap fix hawaii --type sequential --out hawaii-fixed.png``
   * - ``scicomap cvd``
     - Generate color-vision-deficiency preview.
     - ``scicomap cvd hawaii --type sequential --out hawaii-cvd.png``
   * - ``scicomap apply``
     - Apply a colormap to an image file.
     - ``scicomap apply thermal --type sequential --image input.png --out output.png``
   * - ``scicomap wizard``
     - Guided inspection with optional stages.
     - ``scicomap wizard --type sequential --cmap thermal --no-interactive``
   * - ``scicomap report``
     - One-command report bundle (JSON + images + summary).
     - ``scicomap report --cmap hawaii --type sequential --out reports/hawaii``
   * - ``scicomap doctor``
     - Environment and path diagnostics.
     - ``scicomap doctor --json``

Image handling
--------------

``apply``, wizard apply workflows, and reports using an image file share the
same conversion modes. ``luminance`` combines RGB channels using weights
0.2126, 0.7152, and 0.0722; ``first-channel`` uses the red channel;
``gray-only`` requires a single-channel image. Scalar values are scaled from
the image minimum and maximum to [0, 1]; constant or nearly constant images
use 0.
Input alpha is preserved when saving to a format that supports transparency,
such as PNG. Reports using a builtin image still produce a rendered figure.
Unreadable or malformed images produce an actionable error.

``doctor`` checks directory writability with a temporary file that is removed
after the check, preserving existing files.

Workflow stages and diagnostics
-------------------------------

Inspection is the default. Wizard and report run corrections only with
``--fix``, simulations only with ``--cvd``, and image application only with
``--apply``. Supplying ``--image`` or ``--lightness-rounding`` does not enable
a stage. ``--no-fix``, ``--no-cvd``, and ``--no-apply`` disable each stage.
The lightness rounding option has the same meaning as the Python parameter:
it rounds the lower bound in CAM02-UCS J' units, rather than adding lightness.

Only wizard prompts. It offers optional stages with a default of no, and
collects missing image and output paths before validation. ``--no-interactive``
disables its prompts; ``--json`` always disables prompts and figure display.
Without rendering options, ``scicomap wizard --json`` returns diagnostics only.
Reports always write a bundle; wizard's ``--out`` is an image file.

Diagnostics, CVD simulations, and applied images use the corrected map when
fix is enabled, and the original otherwise. Reports retain an original
assessment alongside the corrected assessment and identify the selected map
in ``map_used``. The ``diagnostics`` field describes that selected map;
``original_diagnostics`` describes the original.
``transformed_diagnostics`` describes the correction, or is null if disabled.
The text summary shows both stages and each artifact's map.

``fix`` and wizard support ``--export path.json`` for reusable color tables;
wizard requires ``--fix``. Exporting alone does not render a figure or require
``--out``. Report bundles with ``--fix`` include ``corrected-cmap.json``.
All commands that take maps accept exported ``.json`` files. See
:doc:`user-guide` for the format, loading examples, and scalar normalization.

Statuses are heuristics, not accessibility certification. Sequential maps are
checked for lightness progression; diverging maps for progression on each
branch toward a central extremum; multi-sequential maps for progression within
each half. Circular maps are checked for an endpoint seam and repeated
lightness oscillations. Unordered qualitative and miscellaneous maps receive
a caution to inspect color distinctions in context, rather than a request to
linearize their lightness. ``monotonic_lightness`` is a measurement of the whole
map, so a false value alone does not indicate a problem for every family.

CVD images simulate selected vision conditions. They do not establish that
colors are distinguishable for every viewer or certify a figure's accessibility.
``cvd_simulation`` records the selected map, sample count, and Colorspacious
model conditions: deuteranomaly at severity 50 and 100, protanomaly at 50,
and tritanomaly at 100, with simulated RGB clipped to [0, 1].

Output modes
------------

Every command accepts ``--json``. Both output modes run the same operations.
Rendering in JSON mode requires an explicit ``--out`` file, or a directory
for report. Human figure commands without ``--out`` display the figure.

JSON responses share ``ok``, ``command``, ``inputs``, ``data``, ``warnings``,
and ``errors``. ``list`` returns families as an array and counts as an object.
Rendered outputs use ``data.artifacts`` records containing ``kind``, an
absolute ``path``, and the ``map`` used (``original`` or ``transformed``).
Human figure display uses the path value ``displayed``. A report's stored
JSON matches its emitted JSON, including paths to the report and summary.

Exit codes are 0 for success, 2 for invalid arguments or inputs, and 1 for
operational failures such as an unwritable destination. JSON errors use the
same response fields, including for argument parsing failures.

Learn by example
----------------

- Full notebook walkthrough: :doc:`notebooks/tutorial`
- Interactive command-to-figure exploration: :doc:`tutorial-marimo`

Equivalent workflow in Python API
---------------------------------

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         import scicomap as sc

         cmap = sc.ScicoSequential(cmap="hawaii")
         cmap.assess_cmap(figsize=(14, 6))
         cmap.unif_sym_cmap(lightness_rounding=None, bitonic=False, diffuse=True)
         cmap.assess_cmap(figsize=(14, 6))

   .. tab:: CLI

      .. code-block:: shell

         scicomap check hawaii --type sequential
         scicomap fix hawaii --type sequential --out hawaii-fixed.png

What command outputs look like
------------------------------

.. figure:: pics/hawaii-examples.png
   :width: 78%
   :alt: Assessment-style output examples from scicomap workflows.

   ``check``/``preview``/``compare`` style workflows produce artifact-oriented
   visual diagnostics.

.. figure:: pics/hawaii-fixed-examples.png
   :width: 78%
   :alt: Fixed-output examples after uniformization.

   ``fix`` and ``report`` workflows help produce smoother gradients and clearer
   transitions in real plots.

.. figure:: pics/color-def.png
   :width: 58%
   :alt: Color-vision deficiency concept image.

   ``cvd`` simulates how your map may appear under common color-vision
   deficiency conditions.
