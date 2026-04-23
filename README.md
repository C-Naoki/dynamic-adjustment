# Modeling Covariate Transition for Efficient Estimation of Longitudinal Treatment Effects in Randomized Experiments

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31113/)
[![Pyenv](https://img.shields.io/badge/Pyenv-2.6.7-yellow.svg)](https://github.com/pyenv/pyenv#installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

This repository contains the implementation of "Modeling Covariate Transition for Efficient Estimation of Longitudinal Treatment Effects in Randomized Experiments."

## Usage
1. Clone this repository.
    ```bash
    git clone https://github.com/C-Naoki/DynamicRA.git
    ```
2. Construct a virtual environment and install the required packages.
    ```bash
    make install
    ```
    - Note that it requires [pyenv](https://github.com/pyenv/pyenv#installation) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
    - If you prefer not to use them, you can also use [`requirements.txt`](https://github.com/C-Naoki/DynamicRA/blob/main/requirements.txt) created based on [`pyproject.toml`](https://github.com/C-Naoki/DynamicRA/blob/main/pyproject.toml).

    Specifically, the above command performs the following steps:
    1. if necessary, install Python 3.11.13 using pyenv, and then switch to this version.
    2. create a virtual environment based on Python 3.11.13
    3. install packages in [`pyproject.toml`](https://github.com/C-Naoki/DynamicRA/blob/main/pyproject.toml).
    4. attach the path file (i.e., `*.pth`) in the `site-packages/` for extending module search path.

    Please check the [`Makefile`](https://github.com/C-Naoki/DynamicRA/blob/main/Makefile) for more details.

3. Run quick demos of DynamicRA
    ```bash
    bash bin/demo.sh
    ```
    If you want the command to continue running after logging out, you use `-n` option as shown below (using nohup).
    ```bash
    bash bin/demo.sh -n
    ```
    - The execution log is saved in `nohup/` directory.
