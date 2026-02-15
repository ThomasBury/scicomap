Getting Started
===============

Install
-------

.. code-block:: shell

   pip install -U scicomap

Quickstart
----------

The example below creates a colormap helper, inspects one colormap, and draws
an artifact-focused example.

.. code-block:: python

   import scicomap as sc

   cmap = sc.ScicoSequential(cmap="hawaii")
   cmap.assess_cmap(figsize=(14, 6))
   cmap.draw_example()

Choose a colormap family
------------------------

.. code-block:: python

   sc_map = sc.SciCoMap()
   sc_map.get_ctype()

Typical output:

.. code-block:: text

   dict_keys(['diverging', 'sequential', 'multi-sequential', 'circular', 'miscellaneous', 'qualitative'])

Get a Matplotlib colormap object
--------------------------------

.. code-block:: python

   plt_cmap_obj = cmap.get_mpl_color_map()

Where to go next
----------------

- Read :doc:`user-guide` for common workflows.
- Open :doc:`notebooks/tutorial` for the complete walkthrough.
- Try :doc:`tutorial-marimo` for an interactive browser tutorial.
- Check :doc:`faq` for practical decision rules.
