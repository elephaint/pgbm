# PGBM <img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="300" alt="Airlab Amsterdam" align="right"> #
[![PyPi version](https://img.shields.io/pypi/v/pgbm)](https://pypi.org/project/pgbm/)
[![Python version](https://img.shields.io/pypi/pyversions/pgbm)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub license](https://img.shields.io/pypi/l/pgbm)](https://github.com/elephaint/pgbm/blob/main/LICENSE)

_Probabilistic Gradient Boosting Machines_ (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch/Numba, developed by Airlab in Amsterdam. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_bostonhousing_cpu.py))
* Auto-differentiation of custom loss functions. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example08_bostonhousing_autodiff.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example10_covidhospitaladmissions.py))
* Native GPU-acceleration. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_bostonhousing_gpu.py))
* Distributed training for CPU and GPU, across multiple nodes. ([examples](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/))
* Ability to optimize probabilistic estimates after training for a set of common distributions, without retraining the model. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example07_optimizeddistribution.py))

In addition, we support the following features:
* Feature subsampling by tree
* Sample subsampling ('bagging') by tree
* Saving, loading and predicting with a trained model ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example11_bostonhousing_saveandload.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example11_bostonhousing_saveandload.py))
* Checkpointing (continuing training of a model after saving) ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example12_bostonhousing_checkpointing.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example12_bostonhousing_checkpointing.py))
* Feature importance by gain and permutation ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example09_bostonhousing_featimportance.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example09_bostonhousing_featimportance.py))
* Monotone constraints ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example15_monotone_constraints.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example13_monotone_constraints.py))

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

### Installation ###
Run `pip install pgbm` from a terminal within a Python (virtual) environment of your choice.

#### Verification ####
* Download & run an example from the examples folder to verify the installation is correct:
  * Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_bostonhousing_cpu.py) to verify ability to train & predict on CPU with Torch backend.
  * Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_bostonhousing_gpu.py) to verify ability to train & predict on GPU with Torch backend.
  * Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example01_bostonhousing_cpu.py) to verify ability to train & predict on CPU with Numba backend.
  * Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example13_boston_dist.py) to verify ability to perform distributed CPU, GPU, multi-CPU and/or multi-GPU training.
* Note that when training on the GPU, the custom CUDA kernel will be JIT-compiled when initializing a model. Hence, the first time you train a model on the GPU it can take a bit longer, as PGBM needs to compile the CUDA kernel. 
* When using the Numba-backend, several functions need to be JIT-compiled. Hence, the first time you train a model using this backend it can take a bit longer.
* To run the examples some additional packages such as `scikit-learn` or `matplotlib` are required; these should be installed separately via `pip` or  `conda`.

#### Dependencies ####
The core package has the following dependencies which should be installed separately (installing the core package via `pip` will not automatically install these dependencies).

##### Torch backend #####
* CUDA Toolkit matching your PyTorch distribution (https://developer.nvidia.com/cuda-toolkit)
* PyTorch >= 1.8.0, with CUDA 10.2 for GPU acceleration (https://pytorch.org/get-started/locally/). Verify that PyTorch can find a cuda device on your machine by checking whether `torch.cuda.is_available()` returns `True` after installing PyTorch.
* PGBM uses a custom CUDA kernel which needs to be compiled, which may require installing a suitable compiler. Installing PyTorch and the full CUDA Toolkit should be sufficient, but [open an issue](https://github.com/elephaint/pgbm/issues) if you find it still not working even after installing these dependencies. 
* The CUDA device should have CUDA compute ability 6.x or higher.

##### Numba backend #####
* Numba >= 0.53.1 (https://numba.readthedocs.io/en/stable/user/installing.html). 

The Numba backend does not support differentiable loss functions and GPU training is also not supported using this backend.

### Support ###
See the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder for examples, an overview of hyperparameters and a function reference. In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric.

In case further support is required, [open an issue](https://github.com/elephaint/pgbm/issues).

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://arxiv.org/abs/2106.01682). Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ([KDD 21](https://www.kdd.org/kdd2021/)), August 14–18, 2021, Virtual Event, Singapore.

The experiments from our paper can be replicated by running the scripts in the [experiments](https://github.com/elephaint/pgbm/tree/main/paper/experiments) folder. Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to the [datasets](https://github.com/elephaint/pgbm/tree/main/paper/datasets) folder (Higgs) and to datasets/m5 (m5).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://icai.ai/airlab/).