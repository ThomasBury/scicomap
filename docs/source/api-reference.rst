API Reference
=============

This reference lists the public classes and common methods used in daily work.

Core entry point
----------------

``sc.SciCoMap``

- ``get_ctype()``: list available colormap families.
- ``get_color_map_names()``: list colormaps in the selected family.
- ``get_mpl_color_map()``: return a Matplotlib colormap object.
- ``assess_cmap(...)``: inspect lightness, chroma, and colorblind behavior.
- ``unif_sym_cmap(...)``: apply uniformization and optional symmetrization.

Family-specific classes
-----------------------

- ``sc.ScicoSequential``
- ``sc.ScicoDiverging``
- ``sc.ScicoMultiSequential``
- ``sc.ScicoCircular``
- ``sc.ScicoQualitative``
- ``sc.ScicoMiscellaneous``

Each family class supports the same assessment and correction workflow.

Datasets helpers
----------------

Use dataset helpers to quickly reproduce examples:

- ``scicomap.datasets.load_hill_topography()``
- ``scicomap.datasets.load_scan_image()``
- ``scicomap.datasets.load_pic(name=...)``

Colorblind utilities
--------------------

- ``scicomap.cblind.colorblind_vision(...)``
- ``scicomap.cblind.colorblind_transform(...)``

Math and transformation utilities
---------------------------------

Advanced colormap operations are exposed in ``scicomap.cmath`` for custom
processing pipelines.

For executable end-to-end examples, see :doc:`user-guide` and
:doc:`notebooks/tutorial`.
