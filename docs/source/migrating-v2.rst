Migrating to v2
===============

Version 2.0.0 removes ambiguous state and duplicate command surfaces. Update
callers directly; removed interfaces have no forwarding aliases.

Python before and after
-----------------------

Discovery now uses one catalog. Replace ``get_available_ctype``, ``get_ctype``,
``get_color_map_dic``, and ``get_color_map_names`` with ``get_cmap_dict()``.

.. code-block:: text

   # Before (v1)
   chart.get_color_map_names()

.. code-block:: python

   import scicomap as sc

   chart = sc.ScicoSequential("hawaii")
   names = list(sc.get_cmap_dict()[chart.ctype])

Transformations return a Matplotlib colormap directly. Methods also replace
``chart.cmap``. Remove caller-controlled ``uniformized`` arguments, tuple
unpacking, and access to the old ``uniformized`` attribute. Replace ``lift``
with ``lightness_rounding``: it rounds the lower lightness bound up to a
multiple of the step, rather than adding a fixed amount.

.. code-block:: text

   # Before (v1)
   corrected, uniformized = sc.uniformize_cmap(original, lift=10, uniformized=False)

.. code-block:: python

   original = chart.cmap
   corrected = sc.uniformize_cmap(original, lightness_rounding=10)
   current = chart.unif_sym_cmap(lightness_rounding=0, bitonic=False)
   assert current is chart.cmap

Construction resolves and validates the map immediately. ``ScicoMultiSequential``
now defaults to ``bukavu``. The public exports are explicit; import NumPy,
pyplot, and palette providers from their own packages. Dataset and numerical
helpers live in ``sc.datasets``, ``sc.cmath``, and ``sc.cblind``.

Plots return Figures and do not display them implicitly. ``colorblind``
simulates the current map without transforming it first. Standalone plotting
functions retain their explicit correction options; set ``uniformize=False``
and ``symmetrize=False`` to inspect original maps.

.. code-block:: python

   figure = chart.colorblind()
   # Save with figure.savefig("cvd.png"), or display with plt.show().

CLI before and after
--------------------

Use top-level commands. The duplicate ``cmap`` group, ``docs-llm`` and
``docs llm-assets`` commands, profiles, inferred goals, and alternate output
formats are removed. Documentation generation remains a contributor task via
``just docs``. Replace ``--lift`` with ``--lightness-rounding``.

.. code-block:: text

   # Before (v1)
   scicomap cmap fix hawaii --lift 10 --out fixed.png
   scicomap report --cmap hawaii --profile publication --out report

.. code-block:: shell

   scicomap fix hawaii --lightness-rounding 10 --out fixed.png
   scicomap report --cmap hawaii --fix --cvd --out report --json

Inspection is the default. ``--fix``, ``--cvd``, and ``--apply`` request stages
explicitly; supplying an image or rounding step alone enables no stage. Only
``wizard`` prompts. Every command supports ``--json`` with the same envelope;
machine rendering requires an output destination and never opens a window.
Artifact paths are absolute. Exit codes are 0 for success, 2 for invalid inputs,
and 1 for operational failures. See :doc:`cli-reference` for the JSON contract.

Corrections can be exported as JSON color tables and loaded with Matplotlib
``ListedColormap`` or passed to any CLI map argument. Reports distinguish
original and transformed diagnostics and identify the map used by artifacts.
See :doc:`user-guide` for an example using scalar elevation data and for
provenance limits. Diagnostics are heuristics; CVD simulations describe model
conditions and do not certify accessibility.
