# Examples #

This folder contains examples of PGBM. Our examples are aimed to illustrate the following:
* Examples 1-3: How to train PGBM: on CPU, GPU and multi-GPU.
* Example 4: How to train PGBM using a validation loop.
* Examples 5-6: How PGBM compares to other methods such as NGBoost and LightGBM.
* Example 7: How the choice of output distribution can be optimized after training.

# Hyperparameters #
PGBM employs the following set of hyperparameters:
* `min_split_gain`, default = `0.0`. The minimum gain for a node to split when building a tree.
* `min_data_in_leaf`, default= `2`. The minimum number of samples in a leaf of a tree. Increase this value to reduce overfit.
* `max_leaves`, default=`32`. The maximum number of leaves. Increase this value to create more complicated trees, and reduce the value to create simpler trees (reduce overfitting).
* `max_bin`, default=`256`. The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit.
* `learning_rate`, default=`0.1`. The learning rate of the algorithm; the amount of each new tree prediction that should be added to the ensemble.
* `n_estimators`, default=`100`. The number of trees to create. Typically setting this value higher may improve performance, at the expense of training speed and potential for overfit. Use in conjunction with `learning rate` and `max_leaves`; more trees generally requires a lower `learning_rate` and/or a lower `max_leaves`.
* `verbose`, default=`2`. Flag to output training results per iteration. Set to `1` to supress output.
* `early_stopping_rounds`, default = `100`. The number of iterations after which the training stops should the validation metric not improve.
* `feature_fraction`, default=`1`. Fraction of features to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of features to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `bagging_fraction`, default=`1`. Fraction of samples to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of samples to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `seed`, default=`1`. Random seed to use for `feature_fraction` and `bagging_fraction`.
* `lambda`, default=`1`, constraints`>0`. Regularization parameter. 
* `tree_correlation`, default=`0.03`. Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble.
* `device`, default=`cpu`. Traininig device. Choices are `cpu` or `gpu`.
* `output_device`, default=`cpu`. Only applicable when training on `gpu`. When training on `gpu`, it is possible to run everything on `cpu` except for the split decision. In that case, use `gpu` as `device` and `cpu` as `output_device`.
* `gpu_device_ids`, default=`(0,)`. Dictionary containing the indices of the GPUs used for training. To train on multiple GPUs, use e.g. `(0, 1, 2)`.
* `derivatives`, default=`exact`. If a loss function with an analytical gradient and hessian is provided, use `exact`. If a loss function with a scalar, differentiable loss is provided, use `approx` to let PyTorch use auto-differentiation to calculate the gradient and (approximate) hessian.
* `distribution`, default=`normal`. Choice of output distribution. Choices are `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gumbel`, `weibull`, `poisson`, `negativebinomial`]


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
In the latter case, GPUs corresponding to id 1 to 3 will be used, where device with id 1 will be used to return the results to. 