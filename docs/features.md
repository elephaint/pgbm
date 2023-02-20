# Features #
_Probabilistic Gradient Boosting Machines_ (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch/Numba, developed by Airlab in Amsterdam. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates. ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example01_housing_cpu.py))
* Auto-differentiation of custom loss functions. ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example08_housing_autodiff.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example10_covidhospitaladmissions.py))
* Native GPU-acceleration. ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example02_housing_gpu.py))
* Distributed training for CPU and GPU, across multiple nodes. ([examples](https://github.com/elephaint/pgbm/blob/main/examples/torch_dist/))
* Ability to optimize probabilistic estimates after training for a set of common distributions, without retraining the model. ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example07_optimizeddistribution.py))

In addition, we support the following features:
* Feature subsampling by tree
* Sample subsampling ('bagging') by tree
* Saving, loading and predicting with a trained model ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example11_housing_saveandload.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example11_housing_saveandload.py))
* Checkpointing (continuing training of a model after saving) ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example12_housing_checkpointing.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example12_housing_checkpointing.py))
* Feature importance by gain and permutation ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example09_housing_featimportance.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example09_housing_featimportance.py))
* Monotone constraints ([example](https://github.com/elephaint/pgbm/blob/main/examples/torch/example15_monotone_constraints.py), [example](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example13_monotone_constraints.py))
* Scikit-learn compatible via [`PGBMRegressor`](./function_reference.rst) and [`HistGradientBoostingRegressor`](./function_reference.rst)

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

## Python API ##
We expose PGBM as a Python module based on two backends: Torch and Scikit-learn. To import the base PGBM class:
```
# Torch backend (2 estimators)
from pgbm.torch import PGBM # Torch backend
from pgbm.torch import PGBMRegressor # Torch backend, scikit-learn compatible estimator

# Scikit-learn backend
from pgbm.sklearn import HistGradientBoostingRegressor # Scikit-learn backend
```
Both backends are NOT compatible with each other, meaning that a model trained and saved using one backend can NOT be loaded for continual training or predictions in the other backend.

For details on the `PGBM`, `PGBMRegressor` or `HistGradientBoostingRegressor` class, we refer to the [Function reference](./function_reference.rst).

Our Scikit-learn backend is a modified version of [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) and thus should be fully compatible with scikit-learn.

## Feature overview ##
The table below lists the features per API, which may help you decide which API works best for your usecase. A description of the features is given below the table.

| Feature         | pgbm.torch.PGBM | pgbm.sklearn.HistGradientBoostingRegressor |
|-----------------|:---------------:|:------------------------------------------:|
| Backend         |      Torch      |               Scikit-learn                 |
| CPU training    |      Yes        |                   Yes                      |
| GPU training    |      Yes        |                   No                       |
| Sample bagging  |      Yes        |                   No                       |
| Feature bagging |      Yes        |                   No                       |
| Monotone cst    |      Yes        |                   Yes                      |
| Categorical val |      No         |                   Yes                      |
| Missing values  |      Yes        |                   Yes                      |
| Checkpointing   |      Yes        |                   Yes                      |
| Autodiff        |      Yes        |                   No                       |

Description of features:
* CPU training: if the PGBM model can be trained on a CPU.
* GPU training: if the PGBM model can be trained on a CUDA-compatible GPU.
* Sample bagging: if we can train on a subsample of the dataset. This may improve model accuracy and speeds up training. 
* Feature bagging: if we can train on a subsample of the features of the dataset. This may improve model accuracy and speeds up training.
* Monotone cst: if we can set monotone constraints per feature, using positive, negative or neutral constraints.
* Categorical val: if the model can natively handle categorical data.
* Missing values: if the model can natively handle missing values (defined as NaNs). 
* Checkpointing: if we can train the model, save it, and continue training later on (a.k.a., 'warm-start').
* Autodiff: if we can supply a differentiable loss function for which we use autodifferentiation to determine the gradient and hessian.