User Guide
==========

Choose the right colormap type
------------------------------

Use sequential colormaps for ordered values, diverging colormaps for values
around a midpoint, and qualitative colormaps for categories.

If your figure shows directional or cyclic variables (phase, angle), use
circular colormaps.

Assess a colormap before using it
---------------------------------

Use ``assess_cmap`` to inspect lightness progression, chroma behavior, and
colorblind rendering.

.. code-block:: python

   import matplotlib.pyplot as plt
   import scicomap as sc

   jet = plt.get_cmap("jet")
   cmap = sc.ScicoMiscellaneous(cmap=jet)
   cmap.assess_cmap(figsize=(14, 6))

Uniformize a colormap
---------------------

When a colormap contains visible artifacts, apply uniformization and reassess.

.. code-block:: python

   cmap.unif_sym_cmap(lift=None, bitonic=False, diffuse=True)
   cmap.assess_cmap(figsize=(14, 6))

Practical workflow
------------------

1. Start with a colormap family that matches your data semantics.
2. Assess lightness and colorblind behavior.
3. Apply uniformization only when needed.
4. Validate with your real data, not only synthetic examples.
