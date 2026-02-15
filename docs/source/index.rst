scicomap documentation
======================

scicomap helps you choose, assess, and improve scientific colormaps so figures
remain readable, faithful to the data, and safer for color-vision-deficient
readers.

Why this matters
----------------

.. figure:: pics/choosing-cmap.png
   :width: 65%
   :alt: Colormap type decision guide.

   Pick a colormap family that matches your data semantics before styling.

.. figure:: pics/jet2.png
   :width: 70%
   :alt: Jet introduces staircase-like artifacts in smooth data.

   Non-uniform maps such as jet/rainbow can create false boundaries and visual
   artifacts in otherwise smooth fields.

Who this is for
---------------

- Researchers preparing publication figures.
- Data analysts and data scientists building trustworthy dashboards.
- Engineers who need robust colormap defaults in Matplotlib workflows.

Quick start
-----------

Same workflow in both interfaces:

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

Choose your path
----------------

- New user: :doc:`getting-started`
- Practical guidance: :doc:`user-guide`
- Full tutorial notebook: :doc:`notebooks/tutorial`
- Interactive playground: :doc:`tutorial-marimo`
- Visual family browser: :doc:`gallery`
- Full API details: :doc:`api-reference`
- CLI command reference: :doc:`cli-reference`

Common tasks
------------

- Assess a colormap before publication.
- Fix non-uniform lightness and chroma artifacts.
- Validate colorblind accessibility.
- Apply a colormap to your own image data.

Advanced and automation
-----------------------

- One-command workflow reports with `status`, artifacts, and recommendations:
  ``scicomap report ...``.
- Profile-driven defaults for quick decisions:
  ``quick-look``, ``publication``, ``presentation``, ``cvd-safe``, ``agent``.
- Machine-friendly docs and JSON outputs for tooling/LLMs:
  :doc:`llm-access`.

Documentation last change: |today|

.. toctree::
   :maxdepth: 2
   :caption: Start Here

   Introduction
   getting-started

.. toctree::
   :maxdepth: 2
   :caption: How-to Guides

   user-guide

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/tutorial.ipynb
   tutorial-marimo
   gallery

.. toctree::
   :maxdepth: 2
   :caption: Reference and Support

   api-reference
   cli-reference
   faq
   troubleshooting
   llm-access
   contributing
