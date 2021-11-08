# PGBM <img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="300" alt="Airlab Amsterdam" align="right"> #
[![PyPi version](https://img.shields.io/pypi/v/pgbm)](https://pypi.org/project/pgbm/)
[![Python version](https://img.shields.io/pypi/pyversions/pgbm)](https://docs.conda.io/en/latest/miniconda.html)
[![GitHub license](https://img.shields.io/pypi/l/pgbm)](https://github.com/elephaint/pgbm/blob/main/LICENSE)

_Probabilistic Gradient Boosting Machines_ (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch/Numba, developed by Airlab in Amsterdam. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_housing_cpu.py))
* Auto-differentiation of custom loss functions. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example08_housing_autodiff.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example10_covidhospitaladmissions.py))
* Native GPU-acceleration. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_housing_gpu.py))
* Distributed training for CPU and GPU, across multiple nodes. ([examples](https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/))
* Ability to optimize probabilistic estimates after training for a set of common distributions, without retraining the model. ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example07_optimizeddistribution.py))

In addition, we support the following features:
* Feature subsampling by tree
* Sample subsampling ('bagging') by tree
* Saving, loading and predicting with a trained model ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example11_housing_saveandload.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example11_housing_saveandload.py))
* Checkpointing (continuing training of a model after saving) ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example12_housing_checkpointing.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example12_housing_checkpointing.py))
* Feature importance by gain and permutation ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example09_housing_featimportance.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example09_housing_featimportance.py))
* Monotone constraints ([example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example15_monotone_constraints.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example13_monotone_constraints.py))
* Scikit-learn compatible via `PGBMRegressor` class. 

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

Below a simple example using our sklearn wrapper:
```
from pgbm import PGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = PGBMRegressor().fit(X_train, y_train)  
yhat_point = model.predict(X_test)
yhat_dist = model.predict_dist(X_test)
```

### Installation ###

#### Dependencies ####
We offer PGBM using two backends, PyTorch (`import pgbm`) and Numba (`import pgbm_nb`).

##### Torch backend #####
* `torch>=1.8.0`, with CUDA Toolkit >= 10.2 for GPU acceleration (https://pytorch.org/get-started/locally/). Verify that PyTorch can find a cuda device on your machine by checking whether `torch.cuda.is_available()` returns `True` after installing PyTorch.
* `ninja>=1.10.2.2` for compiling the custom c++ extensions.
* GPU training: the CUDA device should have CUDA compute ability 6.x or higher.

##### Numba backend #####
* `numba>=0.53.1` (https://numba.readthedocs.io/en/stable/user/installing.html). 
The Numba backend does not support differentiable loss functions and GPU training is also not supported using this backend.

#### Installation via `pip` ####
We recommend to install PGBM using `pip`.

* __without__ dependencies: `pip install pgbm`. Use this if you have already installed the above dependencies separately.
* __with__ dependencies:
  * Torch CPU+GPU: `pip install pgbm[torch-gpu] --find-links https://download.pytorch.org/whl/cu102/torch_stable.html`
  * Torch CPU-only: `pip install pgbm[torch-cpu]`
  * Numba: `pip install pgbm[numba]`
  * All versions (Torch CPU+GPU and Numba): `pip install pgbm[all] --find-links https://download.pytorch.org/whl/cu102/torch_stable.html`

#### Verification ####
Both backends use JIT-compilation so you incur additional compilation time the first time you use PGBM.

To verify, download & run an example from the examples folder to verify the installation is correct:
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example01_housing_cpu.py) to verify ability to train & predict on CPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/example02_housing_gpu.py) to verify ability to train & predict on GPU with Torch backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/numba/example01_housing_cpu.py) to verify ability to train & predict on CPU with Numba backend.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/example13_housing_dist.py) to verify ability to perform distributed CPU, GPU, multi-CPU and/or multi-GPU training.

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