# TerraSpec

A Python library for processing and analyzing spectral data, with a focus on mineral identification using the Tetracorder algorithm.

## Features

- Spectral library aggregation and management
- Simulated spectra generation
- Tetracorder command library setup and configuration
- Mineral spectral analysis
- Support for common spectral data file formats 
- EMIT (Earth Surface Mineral Dust Source Investigation) data handling


## Installation

1. Clone and set up the emit-utils repository:
```
git clone git@github.com:emit-sds/emit-utils.git
cd emit-utils
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..
```
2. Create and activate a conda environment using the provided `terraspec.yml`:

```
git clone git@github.com:fotxoa-geo/terraspec.git
cd terraspec
conda env create -f terraspec.yml
conda activate terraspec
```

Note, the pinned version of tensorflow used here may not install on all systems - you can likely either use a more recent version of tensorflow, and/or ignore its installation if you do not intend to run simulations up and down through the atmosphere with isofit.