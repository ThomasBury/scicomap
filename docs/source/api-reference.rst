API Reference
=============

Discovery and construction
--------------------------

``scicomap.get_cmap_dict()`` is the discovery API. It returns a fresh nested
mapping of family names to catalog names and Matplotlib colormaps.

.. code-block:: python

   import scicomap as sc

   catalog = sc.get_cmap_dict()
   families = list(catalog)
   names = list(catalog["sequential"])
   chart = sc.SciCoMap(ctype="sequential", cmap="thermal")
   assert chart.get_mpl_color_map() is chart.cmap

``SciCoMap`` resolves and validates inputs at construction. ``cmap`` accepts
a name in the selected family, a Matplotlib ``Colormap``, or a nonempty list
of color names or RGB(A) rows in [0, 1]. Sampled values must be finite.
Omitting ``cmap`` selects the family default. Unknown families and names raise
``ValueError``; unsupported input types raise ``TypeError``.

The family conveniences inherit the same operations:

- ``ScicoSequential`` (``thermal``)
- ``ScicoMultiSequential`` (``bukavu``)
- ``ScicoDiverging`` (``wildfire``)
- ``ScicoCircular`` (``colorwheel``)
- ``ScicoMiscellaneous`` (``turbo``)
- ``ScicoQualitative`` (``glasbey_dark``)

Diagnostics and transformations
-------------------------------

``diagnose_cmap(cmap_obj, ctype)`` returns a JSON-compatible dictionary with
classification, lightness progression, family-specific branch/seam checks,
status, reasons, and a recommendation. These are sampled heuristics, not
accessibility certification. Python, the CLI, and tutorials use this function.

.. code-block:: python

   diagnostics = sc.diagnose_cmap(chart.cmap, chart.ctype)
   original = chart.cmap
   corrected = chart.unif_sym_cmap(lightness_rounding=0, bitonic=False)
   assert corrected is chart.cmap
   standalone = sc.unif_sym_cmap(original, lightness_rounding=0, bitonic=False)

``uniformize_cmap``, ``symmetrize_cmap``, and ``unif_sym_cmap`` return the
resulting Matplotlib colormap, both as methods and as module functions.
Methods replace ``chart.cmap``; module functions leave the input unchanged.
Sample count and alpha are preserved. Uniformization warns and retains sampled
colors when the lightness pattern is unrecognized.

``lightness_rounding`` is a finite nonnegative step in CAM02-UCS lightness
J' units: the lower bound becomes ``ceil(J' / step) * step``. ``None`` and
``0`` leave it unchanged. It is not an additive increase in lightness.
``bitonic`` requires a central chroma extremum; ``diffuse`` smooths chroma.

Plotting
--------

``assess_cmap``, ``illustrate_palettes``, ``colorblind``, and ``draw_example``
return ``matplotlib.figure.Figure`` objects. They do not display windows or
transform the current map. Call ``plt.show()`` explicitly or save the Figure.
``colorblind`` uses the current map, including any prior transformation.
Qualitative examples repeat colors when categories outnumber the supplied
colors; supply a larger palette when every category needs a distinct color.

The standalone ``plot_colormap``, ``plot_colorblind_vision``, ``compare_cmap``,
and ``jch_plot`` also return Figures without display. The first three retain
their explicit ``uniformize`` and ``symmetrize`` options; set both to false to
inspect an unchanged map.

Public exports and removed interfaces
-------------------------------------

The package explicitly exports the classes, catalog discovery, diagnostics,
colormap transformations, plotting functions above, and the ``datasets``,
``cmath``, and ``cblind`` modules. Dataset loaders, numerical array utilities,
and CVD transforms are accessed through those modules. Imported dependencies
such as NumPy, pyplot, and palette providers are not package exports.

M5 removes ``get_available_ctype``, ``get_ctype``, ``get_color_map_dic``, and
``get_color_map_names``. Use the catalog mapping instead. It also removes the
Python ``lift`` keyword, caller-controlled ``uniformized`` keyword, tuple
transformation returns, and the ``uniformized`` object attribute. There are no
forwarding aliases. The CLI's current ``--lift`` option remains until M6.

For runnable workflows, see :doc:`user-guide` and :doc:`notebooks/tutorial`.
