# Examples #

* For examples using the PyTorch backend (`pgbm`), see the [pytorch](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/) folder.
* For examples using the Numba backend (`pgbm_nb`), see the [numba](https://github.com/elephaint/pgbm/blob/main/examples/numba/) folder. The Numba backend does not support autodifferentiation and it supports less distributions.

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
* `max_bin`, default=`256`, constraint`<32,767`. The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit. 
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
PGBM is a lightweight package. These are its core functions (listed in alphabetical order):
* `crps_ensemble(yhat_dist, y)`. Calculate the CRPS score for a set of probabilistic predictions `yhat_dist` and ground truth `y`.
* `load(filename, device)`. Load a model dictionary from a file to a device. The device should be a `torch.device`. 
* `optimize_distribution(X, y, distributions=None, tree_correlations=None)`. Find the distribution and tree correlation that best fits the data according to lowest CRPS score. The parameters `distribution` and `tree_correlation` of a PGBM model will be adjusted to the best values after running this script. This function returns the best found distribution and tree correlation. By default, this function will loop over all available distributions (see Hyperparameters) and over a range of tree correlations in `[0, 0.2]`. A subset of distributions can be chosen by providing these as a list to this function, e.g. `distributions = ['normal', 'logistic']`. For a custom range of tree correlations, supply a range of your choice, e.g. `tree_correlations=np.arange(0, 0.1, step=0.01)` (Numba backend) or `tree_correlations=torch.arange(0, 0.1, step=0.01)` (Torch backend).
* `permutation_importance(X, y=None, n_permutations=10, levels=None)`. Calculate the feature importance by performing permutations across each feature for `n_permutations`. If `y` is given, this function will compute the percentage error for each permutation of the error metric per feature. Hence, the result will tell you how much your error metric will change if that feature is randomly permuted. If `y` is not supplied, this function will return the weighted mean absolute percentage error compared to the base predictions (i.e., the predictions without permuting the features). This function can be slow if there are many features and samples. In addition to this function, one can more easily inspect the feature importance of a PGBM model by using the attribute `.feature_importance`. This feature importance is based on the cumulative split gain computed on the training set during training. Note that permutation importance often provides better results. For a more detailed discussion, see [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py). See also [this example](https://github.com/elephaint/pgbm/blob/main/examples/example09_bostonhousing_featimportance.py), which illustrates both feature importance methods.
* `predict(X, parallel=True)`. Obtain point predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend).
* `predict_dist(X, n_forecasts=100, parallel=True)`. Obtain `n_forecasts` probabilistic predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend).
* `save(filename)`. Save the state dict of a trained model to a file.
* `train(train_set, objective, metric, params=None, valid_set=None, levels=None)`. Train a PGBM model for a given objective and evaluate on a given metric. If no `valid_set` is provided, the learner will train `n_estimators` as set in the `params` dict. For examples of what the objective and metric should look like, see the examples above. For an example of how the `levels` parameter can be used to construct hierarchical forecasts, please see the [hierarchical time series example](https://github.com/elephaint/pgbm/tree/main/paper/experiments/02_hierarchical_time_series) from our paper or the [Covid-19 example](https://github.com/elephaint/pgbm/blob/main/examples/example10_covidhospitaladmissions.py).

# GPU training #
Only applicable for the PyTorch backend. For training on GPU, it is required to set the following hyperparameters:
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