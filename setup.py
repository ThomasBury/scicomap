import os.path
from setuptools import setup, find_packages

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

EXTRAS_REQUIRE = {"tests": ["pytest", "pytest-cov"]}

INSTALL_REQUIRES = [
    "numpy",
    "pandas",
    "scipy",
    "colorspacious",
    "colorcet",
    "cmcrameri",
    "cmocean",
    "cmasher >= 1.5.8",
    "palettable >= 3.3.0",
    "matplotlib >= 3.3.0",
]

KEYWORDS = "color, color map, scientific color maps, uniform "

setup(
    name="scicomap",
    version="0.4.1",
    description="Scientific color maps",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas Bury",
    author_email="bury.thomas@gmail.com",
    packages=find_packages(),
    zip_safe=False,  # the package can run out of an .egg file
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=">=3.6",
    license="MIT",
    keywords=KEYWORDS,
    package_data={'': ['data/*.png', 'data/*.npz', 'data/*.gz', 'data/*.jpg']},
)
