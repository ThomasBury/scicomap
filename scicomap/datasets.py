import gzip
import warnings
import numpy as np
import matplotlib.image as mpimg
from os.path import dirname, join


def load_hill_topography():
    """
    Load hillshading and return elevation
    :return: np.array
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 'jacksboro_fault_dem.npz')
    with np.load(data_file_name) as dem:
        elevation = dem["elevation"]
    return elevation


def load_scan_image():
    """
    Load image of a medical scan
    :return:
    """
    module_path = dirname(__file__)
    data_file_name = join(module_path, 'data', 's1045.ima.gz')
    with gzip.open(data_file_name) as dfile:
        scan_im = np.frombuffer(dfile.read(), np.uint16).reshape((256, 256))
    return scan_im


def load_pic(name="grmhd"):

    if not isinstance(name, str):
        TypeError("name should be a string")

    module_path = dirname(__file__)

    pic_path = join(module_path, 'data', 'grmhd.png')
    if name == "grmhd":
        pic_path = join(module_path, 'data', 'grmhd.png')
    elif name == "vortex":
        pic_path = join(module_path, 'data', 'vortex.jpg')
    elif name == "tng":
        pic_path = join(module_path, 'data', 'tng.jpg')
    else:
        warnings.warn("Using a default image, name should be in ['grmhd', 'vortex', 'tng']")

    img = mpimg.imread(pic_path)
    return img[:, :, 0]
