import gzip
import warnings
from pkg_resources import resource_stream
import numpy as np
import matplotlib.image as mpimg
from os.path import dirname, join


def load_hill_topography():
    """
    Load hillshading and return elevation
    :return: np.array
    """
    stream = resource_stream(__name__, 'data/jacksboro_fault_dem.npz')
    # module_path = dirname(__file__)
    # data_file_name = join(module_path, 'data', 'jacksboro_fault_dem.npz')
    with np.load(stream) as dem:
        elevation = dem["elevation"]
    return elevation


def load_scan_image():
    """
    Load image of a medical scan
    :return:
    """
    # module_path = dirname(__file__)
    # data_file_name = join(module_path, 'data', 's1045.ima.gz')
    stream = resource_stream(__name__, 'data/s1045.ima.gz')
    with gzip.open(stream) as dfile:
        scan_im = np.frombuffer(dfile.read(), np.uint16).reshape((256, 256))
    return scan_im


def load_pic(name="grmhd"):

    if not isinstance(name, str):
        TypeError("name should be a string")

    module_path = dirname(__file__)
    stream = resource_stream(__name__, 'data/grmhd.png')
    # pic_path = join(module_path, 'data', 'grmhd.png')
    if name == "grmhd":
        stream = resource_stream(__name__, 'data/grmhd.png')
    elif name == "vortex":
        stream = resource_stream(__name__, 'data/vortex.jpg')
    elif name == "tng":
        stream = resource_stream(__name__, 'data/tng.jpg')
    else:
        warnings.warn("Using a default image, name should be in ['grmhd', 'vortex', 'tng']")

    img = mpimg.imread(stream)
    return img[:, :, 0]
