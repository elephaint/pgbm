# PGBM <img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="300" alt="Airlab Amsterdam" align="right"> #
[![PyPi version](https://img.shields.io/pypi/v/pgbm)](https://pypi.org/project/pgbm/)
[![Python version](https://img.shields.io/pypi/pyversions/pgbm)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub license](https://img.shields.io/pypi/l/pgbm)](https://github.com/elephaint/pgbm/blob/main/LICENSE)

_Probabilistic Gradient Boosting Machines_ (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch, developed by Airlab in Amsterdam. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates.
* Auto-differentiation of custom loss functions.
* Native GPU-acceleration.

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

### Installation ###
Run `pip install pgbm` from a terminal within the virtual environment of your choice.

#### Verification ####
* Download & run an example from the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder to verify the installation is correct. Use both `gpu` and `cpu` as device to check if you are able to train on both GPU and CPU.
* Note that when training on the GPU, the custom CUDA kernel will be JIT-compiled when initializing a model. Hence, the first time you train a model on the GPU it can take a bit longer, as PGBM needs to compile the CUDA kernel. 
* When using the Numba-backend, several functions need to be JIT-compiled. Hence, the first time you train a model using this backend it can take a bit longer.

#### Dependencies ####
The core package has the following dependencies: 
* PyTorch >= 1.7.0, with CUDA 11.0 for GPU acceleration (https://pytorch.org/get-started/locally/)
* Numpy >= 1.19.2 (install via `pip` or `conda`; https://github.com/numpy/numpy)
* CUDA Toolkit matching your PyTorch distribution (https://developer.nvidia.com/cuda-toolkit)
* PGBM uses a custom CUDA kernel which needs to be compiled, which may require installing a suitable compiler. Installing PyTorch and the full CUDA Toolkit should be sufficient, but contact the author if you find it still not working even after installing these dependencies. 
* To run the experiments comparing against baseline models a number of additional packages may need to be installed via `pip` or  `conda`.

We also provide PGBM based on a Numba backend for those users who do not want to use PyTorch. In that case, it is required to [install Numba](https://numba.readthedocs.io/en/stable/user/installing.html). The Numba backend does not support differentiable loss functions. For an example of using PGBM with the Numba backend, see the [examples](https://github.com/elephaint/pgbm/tree/main/examples). 

### Support ###
See the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder for examples, an overview of hyperparameters and a function reference. In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric.

In case further support is required, [open an issue](https://github.com/elephaint/pgbm/issues).

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://arxiv.org/abs/2106.01682). Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ([KDD ’21](https://www.kdd.org/kdd2021/)), August 14–18, 2021, Virtual Event, Singapore.

The experiments from our paper can be replicated by running the scripts in the [experiments](https://github.com/elephaint/pgbm/tree/main/paper/experiments) folder. Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to the [datasets](https://github.com/elephaint/pgbm/tree/main/paper/datasets) folder (Higgs) and to datasets/m5 (m5).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://icai.ai/airlab/).