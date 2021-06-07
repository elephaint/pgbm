# Examples #

This folder contains examples of PGBM. The examples illustrate the following:
* Examples 1-3: How to train PGBM: on CPU, GPU and multi-GPU.
* Example 4: How to train PGBM using a validation loop.
* Examples 5-6: How PGBM compares to other methods such as NGBoost and LightGBM.
* Example 7: How the choice of output distribution can be optimized after training.
* Example 8: How to use autodifferentiation for loss functions where no analytical gradient or hessian is provided.
* Example 9: How to plot the feature importance of a learner after training.
* Example 10: How we employed PGBM to forecast Covid-19 daily hospital admissions in the Netherlands.
* Example 11: How to save and load a PGBM model. Train and predict using different devices (CPU or GPU) and/or different backends (PyTorch or Numba).
* Example 12: How to use PGBM when using Numba as backend.

Note: to use the `higgs` dataset in any of the examples, download [here](https://archive.ics.uci.edu/ml/datasets/HIGGS), unpack and save `HIGGS.csv` to your local working directory.

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
train_data = (X_train, y_train)
test_data = (X_test, y_test)
# Train on set   
model = pgbm.PGBM()
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions
yhat_point_pgbm = model.predict(test_data[0])
yhat_dist_pgbm = model.predict_dist(test_data[0])
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

# Hyperparameters #
PGBM employs the following set of hyperparameters (listed in alphabetical order):
* `bagging_fraction`, default=`1`. Fraction of samples to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of samples to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `derivatives`, default=`exact`. If a loss function with an analytical gradient and hessian is provided, use `exact`. If a loss function with a scalar, differentiable loss is provided, use `approx` to let PyTorch use auto-differentiation to calculate the gradient and (approximate) hessian. Not applicable for Numba backend.
* `device`, default=`cpu`. Traininig device. Choices are `cpu` or `gpu`. Not applicable for Numba backend.
* `distribution`, default=`normal`. Choice of output distribution for probabilistic predictions. Choices are `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `weibull`, `poisson`, `negativebinomial`. For the Numba backend, only `normal`, `laplace`, `logistic`, `gamma`, `gumbel`, `poisson` are supported.
* `early_stopping_rounds`, default = `100`. The number of iterations after which the training stops should the validation metric not improve. Only applicable in case a validation set is used.
* `feature_fraction`, default=`1`. Fraction of features to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of features to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `gpu_device_ids`, default=`(0,)`. Dictionary containing the indices of the GPUs used for training. To train on multiple GPUs, use e.g. `(0, 1, 2)`. Not applicable for Numba backend.
* `lambda`, default=`1`, constraints`>0`. Regularization parameter. 
* `learning_rate`, default=`0.1`. The learning rate of the algorithm; the amount of each new tree prediction that should be added to the ensemble.
* `max_bin`, default=`256`. The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit.
* `min_data_in_leaf`, default= `2`. The minimum number of samples in a leaf of a tree. Increase this value to reduce overfit.
* `max_leaves`, default=`32`. The maximum number of leaves per tree. Increase this value to create more complicated trees, and reduce the value to create simpler trees (reduce overfitting).
* `min_split_gain`, default = `0.0`. The minimum gain for a node to split when building a tree.
* `n_estimators`, default=`100`. The number of trees to create. Typically setting this value higher may improve performance, at the expense of training speed and potential for overfit. Use in conjunction with `learning rate` and `max_leaves`; more trees generally requires a lower `learning_rate` and/or a lower `max_leaves`.
* `output_device`, default=`cpu`. Only applicable when training on `gpu`. When training on `gpu`, it is possible to run everything on `cpu` except for the split decision. In that case, use `gpu` as `device` and `cpu` as `output_device`. Not applicable for Numba backend.
* `seed`, default=`1`. Random seed to use for `feature_fraction` and `bagging_fraction`.
* `split_parallel`, default=`feature`. Choose from `feature` or `sample`. This parameter determines whether to parallelize the split decision computation across the sample dimension or across the feature dimension. Typically, for smaller datasets with few features `feature` is the fastest, whereas for larger datasets and/or datasets with many (e.g. > 50) features, `sample` will provide better results. Only applicable when using the Numba backend.
* `tree_correlation`, default=`0.03`. Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble. A good starting value is `log_10(n_samples) / 100`.
* `verbose`, default=`2`. Flag to output metric results for each iteration. Set to `1` to supress output.

# Function reference #
PGBM is a lightweight package. These are its core functions:
* `train(train_set, objective, metric, params=None, valid_set=None, levels=None)`. Train a PGBM model for a given objective and evaluate on a given metric. If no `valid_set` is provided, the learner will train `n_estimators` as set in the `params` dict. For examples of what the objective and metric should look like, see the examples above. For an example of how the `levels` parameter can be used to construct hierarchical forecasts, please see the [hierarchical time series example](https://github.com/elephaint/pgbm/tree/main/paper/experiments/02_hierarchical_time_series) from our paper or the [Covid-19 example](https://github.com/elephaint/pgbm/blob/main/examples/example10_covidhospitaladmissions.py).
* `predict(X, parallel=True)`. Obtain point predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend).
* `predict_dist(X, n_forecasts=100, parallel=True)`. Obtain `n_forecasts` probabilistic predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend).
* `crps_ensemble(yhat_dist, y)`. Calculate the CRPS score for a set of probabilistic predictions `yhat_dist` and ground truth `y`.
* `save(filename)`. Save the state dict of a trained model to a file.
* `load(filename, device)`. Load a model dictionary from a file to a device. The device should be a `torch.device`. 
* `permutation_importance(X, y=None, n_permutations=10, levels=None)`. Calculate the feature importance by performing permutations across each feature for `n_permutations`. If `y` is given, this function will compute the percentage error for each permutation of the error metric per feature. Hence, the result will tell you how much your error metric will change if that feature is randomly permuted. If `y` is not supplied, this function will return the weighted mean absolute percentage error compared to the base predictions (i.e., the predictions without permuting the features). This function can be slow if there are many features and samples. In addition to this function, one can more easily inspect the feature importance of a PGBM model by using the attribute `.feature_importance`. This feature importance is based on the cumulative split gain computed on the training set during training. Note that permutation importance often provides better results. For a more detailed discussion, see [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py). See also [this example](https://github.com/elephaint/pgbm/blob/main/examples/example09_bostonhousing_featimportance.py), which illustrates both feature importance methods.

# GPU training #
For training on GPU, it is required to set the following hyperparameters:
```
params['device'] = 'gpu'
params['output_device'] = 'gpu'
```
When training on GPU, PGBM will select the GPU at the first index (0) by default and return the results at that device. This corresponds to the following parameter:
```
params['gpu_device_ids'] = (0,)
```
If one would like to perform multi-gpu training and use different device ids, just list the device ids:
```
params['gpu_device_ids'] = (1, 2, 3)
```
In the latter case, GPUs corresponding to id 1 to 3 will be used, where device with id 1 will be used to return the results to. Note that for multi-gpu training, all data is still loaded onto the GPU with the first id. Only the split decision is parallelized across multiple GPUs when using multiple GPUs. Hence, the package does not yet support full Torch distributed across devices and nodes, but only across multiple devices on the same node.