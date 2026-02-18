import gzip
import warnings
from importlib.resources import files
import numpy as np
import matplotlib.image as mpimg


def load_hill_topography() -> np.ndarray:
    """
    Load hill topography elevation data.

    This function loads hill topography elevation data from a pre-stored NumPy .npz file.
    The elevation data represents the topography of a geographical region.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing elevation data, representing the topography.

    Notes
    -----
    The elevation data is typically a 2D array representing the elevation (height) of
    different points in the geographical region. It can be used for various geographic
    and geological analyses.

    Example
    -------
    >>> from my_module import load_hill_topography
    >>> elevation_data = load_hill_topography()
    >>> print(elevation_data.shape)  # Print the shape of the loaded elevation data.

    """
    resource = files("scicomap").joinpath("data/jacksboro_fault_dem.npz")
    with resource.open("rb") as stream:
        with np.load(stream) as dem:
            elevation = dem["elevation"]
    return elevation


def load_scan_image() -> np.ndarray:
    """
    Load a scanned image.

    This function loads a scanned image from a pre-stored gzip-compressed binary file.
    The image data is assumed to be in a specific format and is returned as a NumPy array.

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the scanned image data.

    Notes
    -----
    The scanned image data is expected to be a grayscale image, and the function assumes
    that it's stored in a specific binary format. The image is typically a 2D array
    of pixel values representing the scanned content.

    Example
    -------
    >>> from my_module import load_scan_image
    >>> scan_image = load_scan_image()
    >>> print(scan_image.shape)  # Print the shape of the loaded scanned image.

    """
    resource = files("scicomap").joinpath("data/s1045.ima.gz")
    with resource.open("rb") as stream, gzip.open(stream) as dfile:
        scan_im = np.frombuffer(dfile.read(), np.uint16).reshape((256, 256))
    return scan_im


def load_pic(name: str = "grmhd") -> np.ndarray:
    """
    Load an image.

    This function loads an image from a pre-stored image file based on the provided `name`.
    The image data is returned as a NumPy array.

    Parameters
    ----------
    name : str, optional
        The name of the image to load. Default is "grmhd".
        Supported names: "grmhd", "vortex", "tng".

    Returns
    -------
    numpy.ndarray
        A NumPy array containing the image data.

    Raises
    ------
    TypeError
        If the `name` parameter is not a string.

    Notes
    -----
    This function loads an image file based on the provided `name`. It assumes
    that the image files are stored in a specific directory structure.

    Example
    -------
    >>> from my_module import load_pic
    >>> grmhd_image = load_pic("grmhd")
    >>> print(grmhd_image.shape)  # Print the shape of the loaded image.

    """

    if not isinstance(name, str):
        raise TypeError("name should be a string")

    resource_name = "data/grmhd.png"
    if name == "grmhd":
        resource_name = "data/grmhd.png"
    elif name == "vortex":
        resource_name = "data/vortex.jpg"
    elif name == "tng":
        resource_name = "data/tng.jpg"
    else:
        warnings.warn(
            "Using a default image, name should be in ['grmhd', 'vortex', 'tng']"
        )

    resource = files("scicomap").joinpath(resource_name)
    with resource.open("rb") as handle:
        img = mpimg.imread(handle)
    return img[:, :, 0]
