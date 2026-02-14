<img src="pics/logo.png" alt="drawing" width="200"/>

[buy me caffeine](https://ko-fi.com/V7V72SOHX)

# Scientific color maps

Scicomap helps you choose, assess, and improve scientific colormaps so your
figures remain readable and faithful to the underlying data.

## Blog post

[Scicomap Medium blog post (free)](https://towardsdatascience.com/your-colour-map-is-bad-heres-how-to-fix-it-lessons-learnt-from-the-event-horizon-telescope-b82523f09469)

[Official Documentation](https://thomasbury.github.io/scicomap/)

[Tutorial notebook](./docs/source/notebooks/tutorial.ipynb)

## Installation

```shell
pip install scicomap
```

## Quickstart

```python
import scicomap as sc

cmap = sc.ScicoSequential(cmap="hawaii")
cmap.assess_cmap(figsize=(14, 6))
cmap.draw_example()
```

## CLI quickstart

```shell
# list families and colormaps
scicomap list
scicomap list sequential

# diagnose and preview
scicomap check hawaii
scicomap preview hawaii --type sequential --out hawaii-assess.png

# guided workflow and environment checks
scicomap wizard
scicomap doctor --json

# compare and fix
scicomap compare hawaii viridis thermal --type sequential --out compare.png
scicomap fix hawaii --type sequential --out hawaii-fixed.png

# apply a colormap to your own image
scicomap apply thermal --type sequential --image input.png --out output.png

# explicit long-form aliases for automation
scicomap cmap assess --cmap hawaii --type sequential --out hawaii-assess.png
scicomap docs llm-assets --html-dir docs/build/html

# one-command workflow report bundle
scicomap report --cmap hawaii --type sequential --out reports/hawaii
scicomap report --cmap thermal --image input.png --goal apply --format json

# profile-driven defaults
scicomap report --profile publication --cmap hawaii
scicomap report --profile cvd-safe --cmap thermal --format json
scicomap wizard --profile quick-look
```

### CLI profiles

- `quick-look`: fast diagnosis, minimal artifacts
- `publication`: quality-first defaults (`improve` + fix + CVD checks)
- `presentation`: publication defaults with brighter lift bias
- `cvd-safe`: accessibility-first, CVD checks enforced
- `agent`: deterministic machine mode (`--format json`, non-interactive)

### Profile precedence

Configuration resolution order:

1. Profile defaults
2. Context inference (for example, image presence)
3. Explicit user flags
4. Strict profile enforcement (`cvd-safe`, `agent`)

## Documentation map

- [Getting Started](https://thomasbury.github.io/scicomap/getting-started.html): install and first workflow
- [User Guide](https://thomasbury.github.io/scicomap/user-guide.html): choosing, assessing, and correcting colormaps
- [API Reference](https://thomasbury.github.io/scicomap/api-reference.html): module and class reference
- [FAQ](https://thomasbury.github.io/scicomap/faq.html) and [Troubleshooting](https://thomasbury.github.io/scicomap/troubleshooting.html): practical answers for common issues
- [LLM Access](https://thomasbury.github.io/scicomap/llm-access.html): `llms.txt` and markdown mirror policy

## Development

Use `uv` for local development and dependency synchronization.
Notebook docs rendered with `nbsphinx` require a `pandoc` binary.

```shell
# create/update the lockfile
uv lock

# create the virtual environment and install project + extras
uv sync --extra lint --extra test --extra docs

# run commands in the project environment
uv run python -m pytest
uv run ruff check src tests
uv run ruff format --check src tests

# build web docs + LLM assets
uv run sphinx-build -n -b html docs/source docs/build/html
uv run python scripts/build_llm_assets.py
```

`Read the Docs` is kept as a temporary fallback during the Pages rollout.

Contribution guidelines are available in `CONTRIBUTING.md`.
Release notes are tracked in `CHANGELOG.md` and GitHub releases.

## Introduction

Scicomap is a package that provides scientific color maps and tools to standardize your favourite color maps if you don't like the built-in ones.
Scicomap currently provides sequential, bi-sequential, diverging, circular, qualitative and miscellaneous color maps. You can easily draw examples, compare the rendering, see how colorblind people will perceive the color maps. I will illustrate the scicomap capabilities below.

This package is heavily based on the [Event Horizon Telescope Plot package](https://github.com/liamedeiros/ehtplot/tree/docs) and uses good color maps found in the [the python portage of the Fabio Crameri](https://github.com/callumrollo/cmcrameri), [cmasher](https://cmasher.readthedocs.io/), [palettable](https://jiffyclub.github.io/palettable/), [colorcet](https://colorcet.holoviz.org/) and [cmocean](https://matplotlib.org/cmocean/)

## Motivation

The accurate representation of data is essential. Many common color maps distort data through uneven colour gradients and are often unreadable to those with color-vision deficiency. An infamous example is the `jet` color map. These color maps do not render all the information you want to illustrate or even worse render false information through artefacts. Scientist or not, your goal is to communicate visual information in the most accurate and appealing fashion. Moreover, do not overlook colour-vision deficiency, which represents 8% of the (Caucasian) male population.

## Color spaces

Perceptual uniformity is the idea that Euclidean distance between colors in color space should match human color perception distance judgements. For example, a blue and red that are at a distance d apart should look as discriminable as green and purple that are at a distance d apart.
Scicomap uses the CAM02-UCS color space (Uniform Colour Space). Its three coordinates are usually denoted by J', a', and b'. And its cylindrical coordinates are J', C', and h'. The perceptual color space Jab is similar to Lab. However, Jab uses an updated color appearance model that in theory provides greater precision for discriminability measurements.

- Lightness: also known as value or tone, is a representation of a color's brightness
- Chroma: the intrinsic difference between a color and gray of an object
- Hue: the degree to which a stimulus can be described as similar to or different from stimuli that are described as red, green, blue, and yellow

## Encoding information

- Lightness J': for a scalar value, intensity. It must vary linearly with the physical quantity
- hue h' can encode an additional physical quantity, the change of hue should be linearly proportional to the quantity. The hue h' is also ideal in making an image more attractive without interfering with the representation of pixel values.
- chroma is less recognizable and should not be used to encode physical information
  
## Color map uniformization

Following the references and the theories, the uniformization is performed by

- Making the color map linear in J'
- Lifting the color map (making it lighter, i.e. increasing the minimal value of J')
- Symmetrizing the chroma to avoid further artefacts
- Avoid kinks and edges in the chroma curve
- Bitonic symmetrization or not

# Scicomap

## Choosing the right type of color maps

Scicomap provides a bunch of color maps for different applications. The different types of color map are

```python
import scicomap as sc
sc_map = sc.SciCoMap()
sc_map.get_ctype()
```

```
dict_keys(['diverging', 'sequential', 'multi-sequential', 'circular', 'miscellaneous', 'qualitative'])
```

I'll refer to the [The misuse of colour in science communication](https://www.nature.com/articles/s41467-020-19160-7.pdf) for choosing the right scientific color map

<td align="left"><img src="pics/choosing-cmap.png" width="500"/></td>

## Get the matplotlib cmap

```python
plt_cmap_obj = sc_map.get_mpl_color_map()
```

## Choosing the color map for a given type

Get the color maps for a given type

```python
sc_map = sc.ScicoSequential()
sc_map.get_color_map_names()
```

```
dict_keys(['afmhot', 'amber', 'amber_r', 'amp', 'apple', 'apple_r', 'autumn', 'batlow', 'bilbao', 'bilbao_r', 'binary', 'Blues', 'bone', 'BuGn', 'BuPu', 'chroma', 'chroma_r', 'cividis', 'cool', 'copper', 'cosmic', 'cosmic_r', 'deep', 'dense', 'dusk', 'dusk_r', 'eclipse', 'eclipse_r', 'ember', 'ember_r', 'fall', 'fall_r', 'gem', 'gem_r', 'gist_gray', 'gist_heat', 'gist_yarg', 'GnBu', 'Greens', 'gray', 'Greys', 'haline', 'hawaii', 'hawaii_r', 'heat', 'heat_r', 'hot', 'ice', 'inferno', 'imola', 'imola_r', 'lapaz', 'lapaz_r', 'magma', 'matter', 'neon', 'neon_r', 'neutral', 'neutral_r', 'nuuk', 'nuuk_r', 'ocean', 'ocean_r', 'OrRd', 'Oranges', 'pink', 'plasma', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'rain', 'rainbow', 'rainbow-sc', 'rainbow-sc_r', 'rainforest', 'rainforest_r', 'RdPu', 'Reds', 'savanna', 'savanna_r', 'sepia', 'sepia_r', 'speed', 'solar', 'spring', 'summer', 'tempo', 'thermal', 'thermal_r', 'thermal-2', 'tokyo', 'tokyo_r', 'tropical', 'tropical_r', 'turbid', 'turku', 'turku_r', 'viridis', 'winter', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd'])
```

## Assessing a color map

In order to assess if a color map should be corrected or not, `scicomap` provides a way to quickly check if the lightness is linear, how asymmetric and smooth is the chroma and how the color map renders for color-deficient users. I will illustrate some of the artefacts using classical images, as the pyramid and specific functions for each kind of color map.

### An infamous example

```python
import scicomap as sc
import matplotlib.pyplot as plt

# the thing that should not be
ugly_jet = plt.get_cmap("jet")
sc_map =  sc.ScicoMiscellaneous(cmap=ugly_jet)
f=sc_map.assess_cmap(figsize=(22,10))
```

<td align="left"><img src="pics/jet.png" width="1000"/></td>

Clearly, the lightness is not linear, has edges and kinks. The chroma is not smooth and asymmetrical. See the below illustration to see how bad and how many artefacts the jet color map introduces

<td align="left"><img src="pics/jet2.png" width="1000"/></td>

## Correcting a color map - Example

### Sequential color map

Let's assess the built-in color map `hawaii` without correction:

```python
sc_map = sc.ScicoSequential(cmap='hawaii')
f=sc_map.assess_cmap(figsize=(22,10))
```

<td align="left"><img src="pics/hawaii.png" width="1000"/></td>

The color map seems ok, however, the lightness is not linear and the chroma is asymmetrical even if smooth. Those small defects introduce artefact in the information rendering, as we can visualize using the following example

```python
f=sc_map.draw_example()
```

<td align="left"><img src="pics/hawaii-examples.png" width="1000"/></td>

We can clearly see the artefacts, especially for the pyramid for which our eyes should only pick out the corners in the pyramid (ideal situation). Those artefacts are even more striking for color-deficient users (this might not always be the case). Hopefully, `scicomap` provides an easy way to correct those defects:

```python
# fixing the color map, using the same minimal lightness (lift=None), 
# not normalizing to bitone and 
# smoothing the chroma
sc_map.unif_sym_cmap(lift=None, 
                     bitonic=False, 
                     diffuse=True)

# re-assess the color map after fixing it                     
f=sc_map.assess_cmap(figsize=(22,10))
```

<td align="left"><img src="pics/hawaii-fixed.png" width="1000"/></td>

After fixing the color map, the artefacts are less present

<td align="left"><img src="pics/hawaii-fixed-examples.png" width="1000"/></td>

# All the built-in color maps

## Sequential

<td align="left"><img src="pics/seq-cmaps-all.png" width="500"/></td>

## Diverging

<td align="left"><img src="pics/div-cmaps-all.png" width="500"/></td>

## Mutli-sequential

<td align="left"><img src="pics/multi-cmaps-all.png" width="500"/></td>

## Miscellaneous

<td align="left"><img src="pics/misc-cmaps-all.png" width="500"/></td>

## Circular

<td align="left"><img src="pics/circular-cmaps-all.png" width="500"/></td>

## Qualitative

<td align="left"><img src="pics/qual-cmaps-all.png" width="500"/></td>

# References

- [The misuse of colour in science communication](https://www.nature.com/articles/s41467-020-19160-7.pdf)
- [Why We Use Bad Color Maps and What You Can Do About It](https://www.kennethmoreland.com/color-advice/BadColorMaps.pdf)
- [THE RAINBOW IS DEAD…LONG LIVE THE RAINBOW! – SERIES OUTLINE](https://mycarta.wordpress.com/2012/05/29/the-rainbow-is-dead-long-live-the-rainbow-series-outline/)
- [Scientific colour maps](https://www.fabiocrameri.ch/colourmaps/)
- [Picking a colour scale for scientific graphics](https://betterfigures.org/2015/06/23/picking-a-colour-scale-for-scientific-graphics/)
- [ColorCET](https://colorcet.com/)
- [Good Colour Maps: How to Design Them](https://arxiv.org/abs/1509.03700)
- [Perceptually uniform color space for image signals including high dynamic range and wide gamut](https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-25-13-15131&id=368272)

# Changes log

### 1.0.0

- Docstring
- Tutorial notebook
- Web documentation

### 0.4

- Including files in source distributions

### 0.3

- Add a section "how to use with matplotlib"
- [Bug] Center diverging color map in examples

### 0.2

- [Bug] Fix typo in chart titles

### 0.1

- First version
