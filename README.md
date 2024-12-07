# Numerical Estimation of the Two-Point Correlation Function in Spin Systems

This repository contains the code, data, and documentation for the thesis **"Numerically Estimating the Two-Point Correlation Function in Spin Systems"** submitted by **Shivom Gupta** to the University of Leipzig. The thesis investigates the spatial correlation functions in the one-dimensional (1D) Ising model using Monte Carlo simulations.

## Overview

The goal of this project is to analyze various estimators for the spatial correlation function in the 1D Ising model and evaluate their performance across different conditions (system size, temperature, and sample size). The repository includes:

- Simulation scripts for generating spin configurations using the Metropolis algorithm.
- Analysis notebooks for computing spatial correlation functions and biases.
- Tools for visualizing autocorrelation and spatial correlation data.
- Latex source files for the thesis.

## Folder Structure

```
├── docs/                       # Additional documentation (if any)
├── latex/                      # Latex files for the thesis
├── plots/                      # Generated plots from the simulations
├── .gitignore                  # Files and folders to be ignored by Git
├── IsingClass.py               # Implementation of the Ising model class
├── acf.py                      # Autocorrelation function calculations
├── acf_curve.ipynb             # Jupyter notebook for plotting autocorrelation curves
├── autocorr.ipynb              # Analysis of temporal correlations
├── corr_funcs.ipynb            # Analysis of spatial correlation functions
├── corr_funcs.py               # Python script for spatial correlation functions
├── data.ipynb                  # Summary of data and preprocessing steps
├── generate_configurations.py  # Monte Carlo simulation script
├── generate_data_server.sh     # Script to automate data generation on a server
├── ising.py                    # Core Ising model simulation logic
├── link_to_data.txt            # Link to large datasets (if hosted externally)
├── profile_ising.py            # Script for profiling the simulation performance
├── random_normal_dist.ipynb    # Analysis of randomness in the system
├── taus.txt                    # Autocorrelation time results
├── test_ising.ipynb            # Notebook for testing the Ising model implementation
```

## Key Features

1. **Monte Carlo Simulations:** Simulate spin configurations using the Metropolis algorithm.
2. **Spatial and Temporal Correlations:** Compute and visualize spatial and temporal correlation functions.
3. **Bias and Variance Analysis:** Compare multiple estimators for the spatial correlation function.
4. **Data Visualization:** Generate intuitive plots to analyze system behavior under varying conditions.

## Usage

### Prerequisites

Ensure you have the following dependencies installed:

- Python (>= 3.9)
- NumPy
- Matplotlib
- Numba
- Jupyter Notebook

### Running Simulations

1. Generate spin configurations:
   ```bash
   python generate_configurations.py
   ```

2. Analyze autocorrelation and spatial correlations:
   - Open `autocorr.ipynb` or `corr_funcs.ipynb` in Jupyter Notebook.
   - Run the cells sequentially.

3. Profile simulation performance:
   ```bash
   python profile_ising.py
   ```

### Accessing Data

If large datasets are hosted externally, refer to the `link_to_data.txt` file for download instructions.

## Thesis

The thesis document provides an in-depth explanation of the methodology, theoretical background, and results. It can be found in the `latex` directory.

You can also access the thesis online via the DOI link:  
[https://doi.org/10.13140/RG.2.2.23969.21608](https://doi.org/10.13140/RG.2.2.23969.21608)

## Citation

If you use this work, please cite it as:

```
Gupta, Shivom. "Numerically Estimating the Two-Point Correlation Function in Spin Systems." University of Leipzig, 2024. DOI: 10.13140/RG.2.2.23969.21608
```
