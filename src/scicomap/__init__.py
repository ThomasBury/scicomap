from . import scicomap
from . import datasets
from . import cmath
from . import cblind

__version__ = "1.0.0"
__all__ = ["scicomap", "datasets", "cmath", "cblind", "utils"]

# bound to upper level
from .scicomap import *
from .datasets import *
from .cmath import *
from .cblind import *


# Author declaration
__author__ = "Thomas Bury"
