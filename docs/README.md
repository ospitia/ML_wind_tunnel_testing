# ML Wind Tunnel Testing

## Overview

This project involves the development of a machine learning solution to optimize the wind tunnel testing process for a vehicle.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Overview Solution Approach](#overview-solution-approach)
- [Modular Implementation](#modular-implementation)
- [References](#references)

## Problem Statement

Developing and evaluating surrogate models for aerodynamic pressure forces using wind tunnel data.

1. Construct a surrogate model for c<sub>total</sub> as a function of H1 and H2.
2. Optimize H1 and H2 for minimum c<sub>total</sub>.
3. Construct a surrogate model for c<sub>total</sub> as a function of surface pressures measurements.
4. Select the optimal subset of pressure sensors.
5. Minimize wind tunnel time for given (H1, H2) points.

## Overview Solution Approach

1. **Data Preparation**: Outlier detection and removal.
2. **Model Training**: Train a Random Forest regressor to predict c<sub>total</sub> based on H1 and H2 or p<sub>1</sub>, ..., p<sub>111</sub>.
3. **Optimization**: Use the `scipy.optimize.minimize` function to find the optimal (H1, H2) that minimize c<sub>total</sub>.
4. **Model Optimization**: Optimize the model by selecting features from the given surface pressure sensors.
5. **Time Optimization**: Find the optimal number of (H1, H2) samples, select the samples using Latin hypercube sampling, and find the optimal sequence of points to minimize time using the route optimization algorithms from [OR-Tools](https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/samples/tsp_cities.py).

## Modular Implementation

The codebase has a modular structure, allowing building blocks to be swapped. For instance, this implementation uses a [scikit-learn](https://scikit-learn.org/stable/) Random Forest regressor, but model modules can be replaced with other types of models, and configurations can be managed from the configuration file and method.

### Dataset + Config

* The script `./wind_tunnel_testing/dataset.py` loads the input data into a dictionary of dataframes and declares initial attribute names.
* The script `./wind_tunnel_testing/config.py` defines paths and constants. It also loads model and project configurations from `./config/config.toml`.

### Features

The script `./wind_tunnel_testing/features.py` includes pipelines to remove outliers, handle models, and make predictions based on the provided data. The script also optimizes input parameters to minimize the aerodynamic pressure measurements and selects a subset of pressure sensors for accurate predictions.

### Modeling + Tests

The script `./wind_tunnel_testing/modeling.py` includes the main methods, building blocks for `./wind_tunnel_testing/features.py`. There are unit tests provided for these methods in `./tests/tests.py` and cover the following aspects:

- Loading the test data.
- Training the Random Forest model.
- Making predictions.
- Optimizing input parameters.
- Selecting features.
- Generating grid sample points.
- Optimizing wind tunnel visit points.

## References

- **[scikit-learn (sklearn)](https://scikit-learn.org/stable/)**: A machine learning library in Python used for implementing the Random Forest regressor.
- **[Google OR-Tools](https://developers.google.com/optimization)**: A library for optimization problems used for solving the optimization tasks in this project.
- **[SciPy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)**: Provides the `scipy.optimize.minimize` function used for optimization.
- **[Latin Hypercube Sampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html)**: A statistical method used for sample generation, available in the `scipy.stats.qmc` module.
