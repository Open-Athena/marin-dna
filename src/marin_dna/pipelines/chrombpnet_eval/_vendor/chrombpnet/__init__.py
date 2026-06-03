"""
ChromBPNet - Chromatin Binding Prediction Network

A PyTorch implementation of ChromBPNet for chromatin binding prediction.
"""

__version__ = "0.0.1"
__author__ = "Lei Xiong"
__email__ = "jsxlei@gmail.com"

# Import main modules
from . import data_utils
from . import chrombpnet
from . import bpnet
from . import model_wrappers
from . import dataset

__all__ = [
    "data_utils",
    "chrombpnet", 
    "bpnet",
    "model_wrappers",
    "dataset",
]
