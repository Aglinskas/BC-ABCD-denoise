# BC-ABCD-denoise

Deep learning-based denoising for ABCD and other fMRI datasets using DeepCorr.

## Description

This package implements a conditional Variational Autoencoder (cVAE) approach for denoising fMRI data, particularly tested on the ABCD (Adolescent Brain Cognitive Development) study datasets. The method uses adversarial training and various loss functions to separate neural signals from noise components.

## Features

- **DeepCorr Models**: Conditional VAE-based denoising architecture
- **Adversarial Training**: Advanced models with adversarial components
- **Multiple Datasets**: Tested on ABCD (A1, A2), Forrest Gump, THINGS, and other datasets
- **Comprehensive Utilities**: Tools for data loading, preprocessing, and visualization
- **Dashboard**: Interactive monitoring and visualization tools

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA-capable GPU (recommended for training)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/BC-ABCD-denoise.git
cd BC-ABCD-denoise

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install from PyPI (once published)

```bash
pip install abcd-denoise
```

### Dependencies

The main dependencies include:
- `torch` - PyTorch for deep learning
- `nibabel` - Neuroimaging file I/O
- `nilearn` - fMRI analysis tools
- `antspyx` - Advanced Normalization Tools
- `numpy`, `scipy` - Numerical computing
- `matplotlib`, `seaborn` - Visualization
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `tqdm` - Progress bars

All dependencies will be automatically installed via pip.

## Usage

### Basic Example

```python
import abcd_denoise
from abcd_denoise import models, utils, datasets

# Load your fMRI data
# ... (specific to your data format)

# Initialize a DeepCorr model
from abcd_denoise.DeepCor_models import cVAE

model = cVAE(
    conf=config,
    in_channels=1,
    in_dim=64,
    latent_dim=(32, 16)
)

# Train the model
# ... (training code)

# Denoise your data
# ... (inference code)
```

### Using Utilities

```python
from abcd_denoise import utils

# Gather NIFTI files
epi, gm, cf, brain, epi_flat, gm_flat, cf_flat = utils.gather_niftis(
    sub='subjectID',
    epi_fn='path/to/epi.nii.gz',
    cf_fn='path/to/cf_mask.nii.gz',
    gm_fn='path/to/gm_mask.nii.gz',
    brain_mask='path/to/brain_mask.nii.gz',
    run=1
)
```

### Working with Notebooks

The `Code/` directory contains numerous Jupyter notebooks demonstrating various analyses:
- `082-DeepCorr-ABCD-face-*.ipynb` - ABCD dataset analysis
- `078-DeepCorr-Forrest-*.ipynb` - Forrest Gump dataset analysis
- `161-refactored-*.ipynb` - Refactored pipeline examples
- `180-refac-*-adversarial-*.ipynb` - Adversarial training examples

## Project Structure

```
BC-ABCD-denoise/
├── Code/                      # Main package code (installed as abcd_denoise)
│   ├── models.py             # Main model definitions
│   ├── DeepCor_models.py     # DeepCorr-specific models
│   ├── utils.py              # Utility functions
│   ├── DeepCor_utils.py      # DeepCorr-specific utilities
│   ├── datasets.py           # Dataset classes
│   ├── dashboard.py          # Visualization dashboard
│   ├── adversarial/          # Adversarial training modules
│   ├── adv_models_*/         # Version-specific adversarial models
│   └── *.ipynb               # Analysis notebooks
├── Misc/                     # Miscellaneous files
├── pyproject.toml            # Package metadata and dependencies
├── MANIFEST.in               # Additional files to include in distribution
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## Testing

This package has been tested on:
- **ABCD Dataset** (A1 and A2 cohorts)
- **Forrest Gump Dataset**
- **THINGS Dataset**
- **ABIDE Dataset**
- **BrainIAK Dataset**

## Development

To contribute or modify the code:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code with black
black Code/

# Run linter
flake8 Code/
```

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ABCD Study (Adolescent Brain Cognitive Development Study)
- All dataset contributors and collaborators
- PyTorch, nibabel, nilearn, and ANTs communities

## Contact

For questions or issues, please open an issue on GitHub or contact the development team.
