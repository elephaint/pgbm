# Features #
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
* Scikit-learn compatible via [`PGBMRegressor`](./function_reference.rst) class. 

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](https://arxiv.org/abs/2106.01682) or check out the [examples](https://github.com/elephaint/pgbm/tree/main/examples).

## Python API ##
We expose PGBM as a Python class based on two backends: Torch and Numba. To import the base PGBM class:
```
from pgbm import PGBM # Torch backend
from pgbm_nb import PGBM # Numba backend
```

Both backends are compatible with each other, meaning that a model trained and saved using one backend can be loaded for continual training or predictions in the other backend.

For details on the PGBM class, we refer to the [Function reference](./function_reference.rst).

## Scikit-learn API ##
We also expose PGBM via the Scikit-learn API based on the two backends: Torch and Numba. To import the PGBMRegressor class:
```
from pgbm import PGBMRegressor # Torch backend
from pgbm_nb import PGBMRegressor # Numba backend
```

Both backends are compatible with each other, meaning that a model trained and saved using one backend can be loaded for continual training or predictions in the other backend.

For details on the PGBMRegressor class, we refer to the [Function reference](./function_reference.rst).