# Modeling Covariate Transition for Efficient Estimation of Longitudinal Treatment Effects in Randomized Experiments

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31113/)
[![Pyenv](https://img.shields.io/badge/Pyenv-2.6.7-yellow.svg)](https://github.com/pyenv/pyenv#installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

This repository contains the implementation of "Modeling Covariate Transition for Efficient Estimation of Longitudinal Treatment Effects in Randomized Experiments," Naoki_Chihara, Tatsushi Oka, Yasuko Matsubara, Yasushi Sakurai, Shota Yasui.
The 43rd International Conference on Machine Learning, [ICML2026](https://icml.cc/).

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

## Minimal Example
```python
import numpy as np

from src.models.dynamicra.model import DynamicRAEstimator
from src.utils import build_arrays


X, Y, W, T = build_arrays(data)
estimator = DynamicRAEstimator(
    method='dynamicra',
    regmodel_name='ols',
    transmodel_name='var',
)

results = estimator.estimate(
    X=X,
    Y=Y,
    W=W,
    times=list(range(T)),
    treatment_path=np.ones(T),
    control_path=np.zeros(T),
    variance_type='moment',
)

for result in results:
    print(
        f"t={result['time']}: "
        f"ATE={result['mu']:.3f}, "
        f"95% CI=({result['ci_lower']:.3f}, {result['ci_upper']:.3f})"
    )
```

## Input Data Format
It expects a long-format `pandas.DataFrame` with the following columns:

| Column | Default name | Type | Description |
| --- | --- | --- | --- |
| Unit ID | `id` | int | Experimental unit identifier. |
| Time | `date` | int | Integer time index from `0` to `T - 1`. |
| Feature | `feature` | str | Variable name, e.g., `Y`, `W`, `X1`, ... |
| Value | `value` | float | Numeric value of that feature. |

Required features are:
- `Y`: outcome.
- `W`: treatment assignment, encoded as integer arms such as `0` and `1`.
- Covariates: all non-`Y`/`W` features are used.

Example:
```text
id,date,feature,value
1,0,Y,2.31
1,0,W,0
1,0,X1,0.42
1,0,X2,-1.10
1,1,Y,2.76
1,1,W,1
1,1,X1,0.50
1,1,X2,-0.87
```

## Citation
If you use this code for your research, please consider citing our paper.

```bibtex
```
