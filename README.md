# Modules
## pyphi Release 4.0
Phi toolbox for multivariate analysis by Sal Garcia (salvadorgarciamunoz@gmail.com, sgarciam@ic.ac.uk)

# PyPhi - Multivariate Analysis Toolbox

**Version 0.1.0** - Modular Architecture

A comprehensive Python library for chemometrics, multivariate analysis, and model-based diagnostics.

**Original Author:** Salvador Garcia-Munoz (salvadorgarciamunoz@gmail.com, sgarciam@ic.ac.uk)  
**Contributors:** Ethan Lavialle, Carlos Perez-Galvan

📚 **Documentation:** https://salvadorgarciamunoz.github.io/pyphi/index.html

---

## ✨ Features

### Core Multivariate Methods
- **PCA** (Principal Component Analysis) - with missing data handling via NIPALS or NLP
- **PLS** (Partial Least Squares) - univariate and multivariate Y, with cross-validation
- **Advanced PLS variants:**
  - LWPLS (Locally Weighted PLS)
  - MBPLS (Multi-block PLS)
  - LPLS (Linear PLS for materials blending)
  - JRPLS (Joint Range PLS)
  - TPLS (Tensor/Three-way PLS)

### Spectral Preprocessing
- Standard Normal Variate (SNV)
- Savitzky-Golay filtering (smoothing & derivatives)
- Mean centering and autoscaling
- Baseline correction
- Multiplicative Scatter Correction (MSC)

### Diagnostics & Analysis
- Hotelling's T² and SPE statistics
- Variable contributions analysis
- Varimax rotation
- Bootstrap confidence intervals
- VIP (Variable Importance in Projection)
- Polynomial regression with PLS-aided variable selection

### Export & Integration
- gPROMS export for hybrid modeling
- Pyomo adaptation for optimization
- EIOT (Economic Input-Output Table) format
- Categorical data encoding
- Materials data parsing for batch manufacturing

### Visualization
- Rich plotting library (`pyphi.plots`) with Bokeh
- Score plots, loadings, diagnostics, contributions, and more

### Batch Analysis
- Batch alignment and multi-way models (`pyphi.batch`)

---

## 🚀 Installation

### Requirements
- Python ≥ 3.11
- NumPy, SciPy, Pandas, Matplotlib, Bokeh
- Pyomo (for NLP-based missing data algorithms)
- Optional: IPOPT or GAMS (for advanced optimization)

### Quick Install (Recommended)

#### From Local Repository
```bash
# Clone the repository
git clone https://github.com/salvadorgarciamunoz/pyphi.git
cd pyphi

# Install with pip (editable mode for development)
pip install -e .

# Or install with Poetry (recommended for development)
poetry install
```

#### From PyPI (Future)
```bash
# Once published to PyPI
pip install pyphi
```

### Verify Installation
```python
import pyphi as phi
print(phi.__version__)  # Should print: 0.1.0

# Test basic functionality
import numpy as np
X = np.random.randn(50, 10)
Y = np.random.randn(50, 3)
plsobj = phi.pls(X, Y, 3)
print("✓ PyPhi installed successfully!")
```

---

## 📖 Quick Start

### Basic PCA
```python
import pyphi as phi
import numpy as np

# Generate sample data
X = np.random.randn(100, 20)

# Build PCA model with 3 components
pcaobj = phi.pca(X, A=3, cross_val=5)

# Access results
T = pcaobj['T']        # Scores
P = pcaobj['P']        # Loadings
r2x = pcaobj['R2X']    # Variance explained

# Diagnostics
from pyphi import plots as pp
pp.score_scatter(pcaobj, [1, 2])
pp.diagnostics(pcaobj)
```

### Basic PLS
```python
import pyphi as phi
import pandas as pd

# Load data
X = pd.read_excel('features.xlsx')
Y = pd.read_excel('responses.xlsx')

# Build PLS model with cross-validation
plsobj = phi.pls(X, Y, A=3, cross_val=5)

# Make predictions
Ynew = phi.pls_pred(Xnew, plsobj)

# Visualize
from pyphi import plots as pp
pp.predvsobs(plsobj, X, Y)
pp.vip(plsobj)
```

### Spectral Preprocessing
```python
import pyphi as phi

# Preprocess NIR spectra
X_snv = phi.spectra_snv(X)                           # SNV normalization
X_sg, M = phi.spectra_savgol(5, 1, 2, X_snv)        # 1st derivative
X_msc = phi.spectra_msc(X)                           # Scatter correction

# Build model on preprocessed data
plsobj = phi.pls(X_sg, Y, A=3)
```

### Advanced: Multi-block PLS
```python
import pyphi as phi

# Organize data into blocks
XMB = {
    'Process': process_data,
    'Raw Materials': materials_data,
    'Spectroscopy': spectra_data
}

# Build multi-block model
mbpls_obj = phi.mbpls(XMB, Y, A=3)
```

---

## 📂 Package Structure

```
pyphi/
├── src/pyphi/              # Main package
│   ├── __init__.py         # Public API
│   ├── pca.py              # PCA models
│   ├── pls.py              # PLS models
│   ├── advanced_pls.py     # LWPLS, MBPLS, LPLS, JRPLS, TPLS
│   ├── spectra.py          # Spectral preprocessing
│   ├── diagnostics.py      # Diagnostics & analysis
│   ├── exports.py          # Export & format conversion
│   ├── batch.py            # Batch analysis
│   ├── plots.py            # Visualization
│   ├── utils.py            # Utility functions
│   └── _internal.py        # Internal helpers
├── tests/                  # Test suite (269 tests!)
├── examples/               # Usage examples
├── docs/                   # Documentation
└── pyproject.toml          # Package configuration
```

---

## 🔧 Optional Dependencies

### IPOPT (for NLP-based missing data handling)

**Recommended installation:**
```bash
# Using Conda (easiest)
conda install -c conda-forge ipopt

# Or download binaries from:
# https://github.com/coin-or/Ipopt/releases
```

**Alternative: GAMS**
```bash
# If GAMS is installed, PyPhi will use it automatically
# Ensure GAMS executables are in your system PATH
```

### NEOS Server (fallback)
If IPOPT/GAMS are not available, PyPhi will use the NEOS server for optimization:
```python
import os
os.environ['NEOS_EMAIL'] = 'your.email@domain.com'
```

---

## 🧪 Testing

PyPhi includes a comprehensive test suite with **269 tests**:

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_pca.py
pytest tests/test_pls.py
pytest tests/test_spectra.py

# Run with coverage
pytest tests/ --cov=pyphi --cov-report=html
```

---

## 📚 Examples

Explore the `examples/` directory for detailed examples:

- **Basic calculations PCA and PLS/** - Getting started with PCA/PLS
- **Batch analysis/** - Batch process alignment and modeling
- **JRPLS and TPLS/** - Advanced multi-material modeling
- **LPLS/** - Linear PLS for blending applications
- **Multi-block PLS/** - Multi-block data analysis
- **NIR Calibration/** - Spectroscopic calibration
- **Varimax Rotation/** - Factor rotation for interpretation

---

## 🆕 What's New in v0.1.0

This version represents a **complete refactoring** of PyPhi into a modern, modular package:

✅ **Modular architecture** - Clean separation into focused modules  
✅ **269 comprehensive tests** - Full test coverage for reliability  
✅ **pip-installable** - Standard Python packaging  
✅ **Modern tooling** - Poetry, pytest, proper src-layout  
✅ **Improved documentation** - Comprehensive docstrings with examples  
✅ **Library optimizations** - Using scipy.stats, numpy built-ins  
✅ **100% backward compatible** - All existing examples work unchanged  

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 🙏 Acknowledgments

Original development by Salvador Garcia-Munoz.  
Refactoring and modernization by Ethan Lavialle and Carlos Perez-Galvan.

PyPhi builds on the excellent work of the NumPy, SciPy, Pandas, and Pyomo communities.

---

## Optional External Dependencies
- IPOPT as an executable in your system path or GAMS python module or GAMS executable in yoru system path.
  - Windows: ```conda install -c conda-forge IPOPT=3.11.1``` or download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), extract and add the IPOPT\bin folder to your system path or add all files to your working directory.
  - Mac/Linux: ```conda install -c conda-forge IPOPT```, download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), or [Compile using coinbrew](https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW).
  
  - if GAMS is installed, pyphi will run ipopt via GAMS, make sure the GAMS executables are reachable through the system PATH

- If IPOPT is not detected, pyphi will submit the pyomo models to the NEOS server to solve them remotely.
  - To use the NEOS server, the environment variable "NEOS_EMAIL" must be assigned a valid email. This can be done outside of python using set/set/export or use ```import os
  os.environ["NEOS_EMAIL"] = youremail@domain.com```
  in your code.


Run the script '''Example_Script_testing_MD_by_NLP.py''' to verify that pyphi can execute IPOPT

Adding a folder to your system path:
 - Windows: temporary ```set PATH=C:\Path\To\ipopt\bin;%PATH%``` or persistent ```setx PATH=C:\Path\To\ipopt\bin;%PATH%```.
 - Mac/Linux: ```export PATH=/path/to/ipopt:$PATH```, add to .profile/.*rc file to make persistent.
 - Both via Conda: after activating your environment, use ```conda env config vars set``` and your OS-specific set or export command.

 ---

**Made with ❤️ for the chemometrics and data science community**
