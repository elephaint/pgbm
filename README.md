# PGBM #

Probabilistic Gradient Boosting Machines (PGBM) is a probabilistic gradient boosting framework in Python based on PyTorch. It provides the following advantages over existing frameworks:
* Probabilistic regression estimates instead of only point estimates.
* Auto-differentiation of custom loss functions.
* Native GPU-acceleration.

It is aimed at users interested in solving large-scale tabular probabilistic regression problems, such as probabilistic time series forecasting. For more details, read [the paper](arxiv-link).

### Installation ###
* Clone the repository
* Go to {your_installation_directory}/pgbm
* Run a demo from the [demos](https://github.com/elephaint/pgbm/tree/main/demos) folder to verify the installation is correct. Use both 'gpu' and 'cpu' as device to check if you are able to train on both GPU and CPU.
* The first time it may take a bit longer to import pgbm as it relies on JIT compilation for the custom CUDA kernel. 

#### Dependencies ####

* PyTorch >= 1.7.0 with CUDA 11.0 for GPU acceleration (https://pytorch.org/get-started/locally/)
* Numpy >= 1.19.2 (install via `pip` or `conda`; https://github.com/numpy/numpy)
* PGBM uses a custom CUDA kernel which needs to be compiled; this may require installing a suitable compiler (e.g. gcc) although installing PyTorch according to the official docs should install all the required dependencies.
* To run the experiments comparing against baseline models a number of additional packages may need to be installed via `pip` or  `conda`.

### Examples ###
See the [demos](https://github.com/elephaint/pgbm/tree/main/demos) folder for more examples. In general, PGBM works similar to existing gradient boosting packages such as LightGBM or xgboost (and it should be possible to more or less use it as a drop-in replacement), except that it is required to explicitly define a loss function and loss metric.

#### Example 1: default settings ####
Below is a simple example that aims to predict house prices for the [Boston Housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). 

First, we import the necessary packages. Note that the first time this can take longer due to the JIT-compilation of the CUDA-kernel. However, in this simple example we will train on the CPU.
```
import torch
import pgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
```
Second, we define our loss function and evaluation metric. 
* The loss function should consume a torch vector of predictions `yhat` and ground truth values `y` and output the gradient and hessian with respect to `yhat` of the loss function.
* The evaluation metric should consume a torch vector of predictions `yhat` and ground truth values `y`, and output a scalar loss.
```
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y):
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
We can now plot the point and probabilistic predictions:
```
plt.plot(test_data[1], 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.legend()
```
[Insert plot]

#### Example 2: Training on GPU ####
For training on GPU, it is required to set the following default parametes:
```
params['device'] = 'gpu'
params['output_device'] = 'gpu'
```
By default PGBM will train on the GPU at the first index (0) and return the results at that device. This corresponds to the following parameter:
```
params['gpu_device_ids'] = (0,)
```
If one would like to perform multi-gpu training and use different device ids, just list the device ids:
```
params['gpu_device_ids'] = (1, 2, 3)
```
In the latter case, GPUs corresponding to id 1 to 3 will be used, where device with id 1 will be used to return the results to.

### Experiments ###
The experiments from our paper can be replicated by running the scripts in the [experiments](https://github.com/elephaint/pgbm/tree/main/experiments) folder. Datasets are downloaded when needed in the experiments except for higgs and m5, which should be pre-downloaded and saved to the [datasets](https://github.com/elephaint/pgbm/tree/main/datasets) folder.

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression](https://linktopaper). Accepted for publication at [SIGKDD '21](https://www.kdd.org/kdd2021/).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### ToDo ###
We intend to have the package as lightweight as possible.

- [x] ~~Add extreme value distributions such as Gumbel and Weibull to distribution choices.~~
- [x] ~~Remove properscoring dependency (crps_ensemble can be calculated much faster on GPU)~~
- [x] ~~Set default values for learning parameters.~~
- [ ] Support feature explainer
- [ ] Full support of Torch distributed (across multiple GPUs and nodes, now only across multiple GPUs supported).
- [ ] Remove JIT-compilation dependency and offer as an installable package via `pip` or `conda`.