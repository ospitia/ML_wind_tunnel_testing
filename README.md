# ML wind tunnel testing

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is designed for a wind tunnel testing experiment. It focuses on:

**Surrogate Modeling**: Building and evaluating surrogate models based on wind tunnel data.

**Optimization**: Determining optimal configurations for wind tunnel experiments and sensor selection while maintaining efficiency and accuracy.

## Project setting
```Bash
conda create -n ML_wind_tunnel_testing python=3.11
```
```Bash
conda activate ML_wind_tunnel_testing
```
```Bash
pip install -r requirements.txt
```
#### optional for development
```Bash
pre-commit install
```

## Run examples
Ensure you have the data file ./data/raw/ML_test.xlsx and run
```Bash
python examples
```
Reports will be written in ./reports/ML_test.xlsx

## Additional Documentation

For detailed information about the algorithms and optimization techniques used in this project, refer to the [Documentation](./docs/README.md).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
│
├── config
│   └── config.toml    <- Model/project config params.
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── pyproject.toml     <- Project configuration file with package metadata for agentme
│                         and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
├── examples
│   └── wind_tunnel_testing <- Running examples of all functionality.
│
├── tests
│   ├── test_wind_tunnel_testing <- Unit tests for modeling methods.
│   └── conftest.py    <- test config.
│
└── wind_tunnel_testing <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes wind_tunnel_testing a Python module
    │
    ├── config.py      <- Script to define overall paths and config params
    │
    ├── dataset.py     <- Scripts to download or generate data
    │
    ├── features.py    <- Scripts to turn raw data into features for modeling
    │
    └── modeling.py    <- Scripts to train models and then use trained models to make
                          predictions
```

--------
