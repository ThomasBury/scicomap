Getting Started
===============

In five minutes, you should be able to pick a colormap, assess it, and run a
safe default improvement workflow.

Install
-------

.. code-block:: shell

   uv add scicomap


or

.. code-block:: shell

   pip install -U scicomap

Quickstart
----------

The same starter workflow is available in Python and CLI forms.

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         import scicomap as sc

         cmap = sc.ScicoSequential(cmap="hawaii")
         cmap.assess_cmap(figsize=(14, 6))
         cmap.unif_sym_cmap(lift=None, bitonic=False, diffuse=True)
         cmap.draw_example()

   .. tab:: CLI

      .. code-block:: shell

         scicomap check hawaii --type sequential
         scicomap report --profile publication --cmap hawaii --type sequential
         scicomap cvd hawaii --type sequential --out hawaii-cvd.png

Expected result:

.. code-block:: text

   - A diagnostics status (good/caution/fix-recommended)
   - A report directory containing summary.txt and report.json
   - A colorblind preview image at hawaii-cvd.png

.. figure:: pics/hawaii-examples.png
   :width: 78%
   :alt: Example output panels for hawaii before correction.

   Typical visual output from assessment-style workflows.

Simple usage
------------

Use these commands and APIs first if you are new to scicomap.

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         import scicomap as sc

         cmap = sc.ScicoSequential(cmap="hawaii")
         cmap.assess_cmap(figsize=(14, 6))

   .. tab:: CLI

      .. code-block:: shell

         scicomap check hawaii --type sequential
         scicomap preview hawaii --type sequential --out hawaii-assess.png

Choose a colormap family
------------------------

.. tabs::

   .. tab:: Python API

      .. code-block:: python

         sc_map = sc.SciCoMap()
         sc_map.get_ctype()

   .. tab:: CLI

      .. code-block:: shell

         scicomap list

Typical output:

.. code-block:: text

   dict_keys(['diverging', 'sequential', 'multi-sequential', 'circular', 'miscellaneous', 'qualitative'])

Get a Matplotlib colormap object
--------------------------------

.. code-block:: python

   plt_cmap_obj = cmap.get_mpl_color_map()

Advanced next steps
-------------------

Use profiles and guided workflows when you want repeatable quality checks.

.. code-block:: shell

   scicomap wizard --profile quick-look --type sequential --cmap thermal --no-interactive
   scicomap report --profile cvd-safe --cmap thermal --format json

.. figure:: pics/hawaii-fixed-examples.png
   :width: 78%
   :alt: Example output panels for hawaii after correction.

   After correction, transitions and gradients are typically more stable across
   test images.

Where to go next
----------------

- Read :doc:`user-guide` for common workflows.
- Open :doc:`cli-reference` for command-first usage.
- Open :doc:`notebooks/tutorial` for the complete walkthrough.
- Try :doc:`tutorial-marimo` for an interactive browser tutorial.
- Check :doc:`faq` for practical decision rules.
