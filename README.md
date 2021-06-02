# PGBM <img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="300" alt="Airlab Amsterdam" align="right"> #

Probabilistic Gradient Boosting Machines (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch, developed by Airlab in Amsterdam. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates.
* Auto-differentiation of custom loss functions.
* Native GPU-acceleration.

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [our paper](arxiv-link).

### Installation ###
Run `pip install pgbm` from a terminal within the virtual environment of your choice.

#### Verification ####
* Download & run an example from the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder to verify the installation is correct. Use both `gpu` and `cpu` as device to check if you are able to train on both GPU and CPU.
* Note that when training on the GPU, the custom CUDA kernel will be JIT-compiled when initializing a model. Hence, the first time you train a model on the GPU it can take a bit longer, as PGBM needs to compile the CUDA kernel. 

#### Dependencies ####
The core package has the following dependencies: 
* PyTorch >= 1.7.0, with CUDA 11.0 for GPU acceleration (https://pytorch.org/get-started/locally/)
* Numpy >= 1.19.2 (install via `pip` or `conda`; https://github.com/numpy/numpy)
* CUDA Toolkit 11.0 (or one matching your PyTorch distribution) (https://developer.nvidia.com/cuda-toolkit)
* PGBM uses a custom CUDA kernel which needs to be compiled, which may require installing a suitable compiler. Installing PyTorch and the full CUDA Toolkit should be sufficient, but contact the author if you find it still not working even after installing these dependencies. 
* To run the experiments comparing against baseline models a number of additional packages may need to be installed via `pip` or  `conda`.

We also provide PGBM based on a Numba backend for those users who do not want to use PyTorch. In that case, it is required to install Numba. The Numba backend does not support differentiable loss functions. For an example of using PGBM with the Numba backend, see the examples.

### Examples ###
See the [examples](https://github.com/elephaint/pgbm/tree/main/examples) folder for examples, an overview of hyperparameters and a function reference. In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric.

Below is an example of a probabilistic regression task: predict housing prices for the [Boston Housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). The code for this example can be found [here](https://github.com/elephaint/pgbm/blob/main/examples/example1_bostonhousing.py).

First, we import the necessary packages. In this simple example we will train on the CPU.
```
import torch
from pgbm import PGBM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
```
Second, we define our loss function and evaluation metric. 
* The loss function should consume a torch vector of predictions `yhat` and ground truth values `y` and output the gradient and hessian with respect to `yhat` of the loss function. For more complicated loss functions, it is possible to add a `levels` variable, but this can be set to `None` in case it is not required.
* The evaluation metric should consume a torch vector of predictions `yhat` and ground truth values `y`, and output a scalar loss. For more complicated evaluation metrics, it is possible to add a `levels` variable, but this can be set to `None` in case it is not required.
```
def mseloss_objective(yhat, y, levels=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y, levels=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
```
Third, we load our data:
```
X, y = load_boston(return_X_y=True)
``` 
Finally, we train our model:
```
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Build tuples of torch datasets
torchdata = lambda x : torch.from_numpy(x).float()
train_data = (torchdata(X_train), torchdata(y_train))
test_data = (torchdata(X_test), torchdata(y_test))
# Train on set   
model = pgbm.PGBM()
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions
yhat_point_pgbm = model.predict(test_data[0])
yhat_dist_pgbm = model.predict_dist(test_data[0], n_samples=1000)
# Scoring
rmse = rmseloss_metric(yhat_point_pgbm, test_data[1])
crps = pgbm.crps_ensemble(test_data[1], yhat_dist_pgbm).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
```
We can now plot the point and probabilistic predictions (indicated by max and min bound on the predictions):
```
plt.plot(test_data[1], 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.legend()
```
which will give us the point forecast and probabilistic forecast:
![Boston Housing probabilistic forecast](/examples/example01_figure.png)

### Support ###
See the [examples](https://github.com/elephaint/pgbm/tree/main/examples) for an overview of hyperparameters and a function reference. [Email the author](mailto:o.r.sprangers@uva.nl) for further support.

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://linktopaper). Accepted for publication at [SIGKDD '21](https://www.kdd.org/kdd2021/).

The experiments from our paper can be replicated by running the scripts in the [experiments](https://github.com/elephaint/pgbm/tree/main/paper/experiments) folder. Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to the [datasets](https://github.com/elephaint/pgbm/tree/main/paper/datasets) folder (Higgs) and to datasets/m5 (m5).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://icai.ai/airlab/).