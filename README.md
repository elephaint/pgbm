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

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. 

For more details, [read the docs](https://pgbm.readthedocs.io/en/latest/index.html) or [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

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

See [Installation](https://pgbm.readthedocs.io/en/latest/installation.html) section in our [docs](https://pgbm.readthedocs.io/en/latest/index.html).

### Support ###
In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric.

* Read the docs for an overview of [hyperparameters](https://pgbm.readthedocs.io/en/latest/parameters.html) and a [function reference](https://pgbm.readthedocs.io/en/latest/function_reference.html).
* See the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder for examples. 

In case further support is required, [open an issue](https://github.com/elephaint/pgbm/issues).

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://arxiv.org/abs/2106.01682). Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining ([KDD 21](https://www.kdd.org/kdd2021/)), August 14–18, 2021, Virtual Event, Singapore.

The experiments from our paper can be replicated by running the scripts in the [experiments](https://github.com/elephaint/pgbm/tree/main/paper/experiments) folder. Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to the [datasets](https://github.com/elephaint/pgbm/tree/main/paper/datasets) folder (Higgs) and to datasets/m5 (m5).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://icai.ai/airlab/).
