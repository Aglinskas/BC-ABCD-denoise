"""
ABCD Denoise - Deep learning-based fMRI denoising using DeepCorr

This package provides tools for denoising fMRI data from ABCD and other datasets
using conditional Variational Autoencoders (cVAE) and adversarial training approaches.
"""

__version__ = "0.1.0"
__author__ = "BC-ABCD-denoise Team"

# Import main modules
try:
    from . import models
    from . import utils
    from . import datasets
    from . import dashboard
    from . import DeepCor_models
    from . import DeepCor_utils
except ImportError:
    # Allow package to be imported even if some modules have issues
    pass

__all__ = [
    "models",
    "utils",
    "datasets",
    "dashboard",
    "DeepCor_models",
    "DeepCor_utils",
]
