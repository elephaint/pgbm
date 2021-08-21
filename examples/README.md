# Examples #

* For examples using the PyTorch backend (`pgbm`), see the [pytorch](https://github.com/elephaint/pgbm/blob/main/examples/pytorch/) folder.
* For examples using the Numba backend (`pgbm_nb`), see the [numba](https://github.com/elephaint/pgbm/blob/main/examples/numba/) folder. The Numba backend does not support autodifferentiation and it supports less distributions.

# Hyperparameters #
PGBM employs the following set of hyperparameters (listed in alphabetical order):
* `bagging_fraction`, default=`1`. Fraction of samples to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of samples to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `checkpoint`, default=`False`. Boolean to save a model checkpoint after each iteration to the current working directory. 
* `derivatives`, default=`exact`. If a loss function with an analytical gradient and hessian is provided, use `exact`. If a loss function with a scalar, differentiable loss is provided, use `approx` to let PyTorch use auto-differentiation to calculate the gradient and (approximate) hessian. Not applicable for Numba backend.
* `device`, default=`cpu`. Traininig device. Choices are `cpu` or `gpu`. Not applicable for Numba backend.
* `distribution`, default=`normal`. Choice of output distribution for probabilistic predictions. Choices are `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `weibull`, `poisson`, `negativebinomial`. For the Numba backend, only `normal`, `studentt`, `laplace`, `logistic`, `gamma`, `gumbel`, `poisson` are supported. Note that the `studentt` distribution has a constant degree-of-freedom of `3`.
* `early_stopping_rounds`, default = `100`. The number of iterations after which the training stops should the validation metric not improve. Only applicable in case a validation set is used.
* `feature_fraction`, default=`1`. Fraction of features to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of features to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `gpu_device_id`, default=`0`. Integer with the index of the GPU used for training. Change this when you'd like to train on a different GPU within your node. For multi-gpu and multinode training, see [here](https://github.com/elephaint/pgbm/blob/main/examples/pytorch_dist/) Not applicable for Numba backend.
* `lambda`, default=`1`, constraints`>0`. Regularization parameter. 
* `learning_rate`, default=`0.1`. The learning rate of the algorithm; the amount of each new tree prediction that should be added to the ensemble.
* `max_bin`, default=`256`, constraint`<32,767`. The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit. 
* `max_leaves`, default=`32`. The maximum number of leaves per tree. Increase this value to create more complicated trees, and reduce the value to create simpler trees (reduce overfitting).
* `min_data_in_leaf`, default= `3`, constraint`>= 3`. The minimum number of samples in a leaf of a tree. Increase this value to reduce overfit.
* `min_split_gain`, default = `0.0`. The minimum gain for a node to split when building a tree.
* `monotone_constraints`, default = `zeros of shape [n_features]`. This allows to provide monotonicity constraints per feature. `1` means increasing, `0` means no constraint, `-1` means decreasing. All features need to be specified when using this parameters, for example `monotone_constraints=[1, 0, -1]` for a positive, non-constraint and negative constraint for respectively feature 1, 2 and 3. There should be limited effect on training speed. To improve accuracy, you can try to increase `monotone_iterataions` (see hereafter), but this comes at the expense of slower training. 
* `monotone_iterations`, default=`1`. The number of alternative splits that will be considered if a monotone constraint is violated by the current split proposal. Increase this to improve accuracy at the expense of training speed.
* `n_estimators`, default=`100`. The number of trees to create. Typically setting this value higher may improve performance, at the expense of training speed and potential for overfit. Use in conjunction with `learning rate` and `max_leaves`; more trees generally requires a lower `learning_rate` and/or a lower `max_leaves`.
* `seed`, default=`2147483647`. Random seed to use for `feature_fraction` and `bagging_fraction` (latter only for Numba backend - for speed considerations the Torch backend `bagging_fraction` determination is not yet deterministic).
* `split_parallel`, default=`feature`. Choose from `feature` or `sample`. This parameter determines whether to parallelize the split decision computation across the sample dimension or across the feature dimension. Typically, for smaller datasets with few features `feature` is the fastest, whereas for larger datasets and/or datasets with many (e.g. > 50) features, `sample` will provide better results. Only applicable when using the Numba backend.
* `tree_correlation`, default=`log_10(n_samples_train) / 100`. Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble. 
* `verbose`, default=`2`. Flag to output metric results for each iteration. Set to `1` to supress output.

# Function reference #
PGBM is a lightweight package. These are its core functions (listed in alphabetical order):
* `crps_ensemble(yhat_dist, y)`. Calculate the CRPS score for a set of probabilistic predictions `yhat_dist` and ground truth `y`.
* `load(filename, device)`. Load a model dictionary from a file to a device. The device should be a `torch.device`. The latter is only applicable for the PyTorch backend. 
* `optimize_distribution(X, y, distributions=None, tree_correlations=None)`. Find the distribution and tree correlation that best fits the data according to lowest CRPS score. The parameters `distribution` and `tree_correlation` of a PGBM model will be adjusted to the best values after running this script. This function returns the best found distribution and tree correlation. By default, this function will loop over all available distributions (see Hyperparameters) and over a range of tree correlations in `[0, 0.2]`. A subset of distributions can be chosen by providing these as a list to this function, e.g. `distributions = ['normal', 'logistic']`. For a custom range of tree correlations, supply a range of your choice, e.g. `tree_correlations=np.arange(0, 0.1, step=0.01)` (Numba backend) or `tree_correlations=torch.arange(0, 0.1, step=0.01)` (Torch backend).
* `permutation_importance(X, y=None, n_permutations=10, levels=None)`. Calculate the feature importance by performing permutations across each feature for `n_permutations`. If `y` is given, this function will compute the percentage error for each permutation of the error metric per feature. Hence, the result will tell you how much your error metric will change if that feature is randomly permuted. If `y` is not supplied, this function will return the weighted mean absolute percentage error compared to the base predictions (i.e., the predictions without permuting the features). This function can be slow if there are many features and samples. In addition to this function, one can more easily inspect the feature importance of a PGBM model by using the attribute `.feature_importance`. This feature importance is based on the cumulative split gain computed on the training set during training. Note that permutation importance often provides better results. For a more detailed discussion, see [here](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py). See also [this example](https://github.com/elephaint/pgbm/blob/main/examples/example09_bostonhousing_featimportance.py), which illustrates both feature importance methods.
* `predict(X, parallel=True)`. Obtain point predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend).
* `predict_dist(X, n_forecasts=100, parallel=True, output_sample_statistics=False)`. Obtain `n_forecasts` probabilistic predictions for a sample set `X`. Use `parallel=False` if you experience out-of-memory errors (only applicable for PyTorch backend). Use `output_sample_statistics=True` to output the tuple `(forecasts, mean, variance)` for each sample.
* `save(filename)`. Save the state dict of a trained model to a file.
* `train(train_set, objective, metric, params=None, valid_set=None, levels=None)`. Train a PGBM model for a given objective and evaluate on a given metric. If no `valid_set` is provided, the learner will train `n_estimators` as set in the `params` dict. For examples of what the objective and metric should look like, see the examples above. For an example of how the `levels` parameter can be used to construct hierarchical forecasts, please see the [hierarchical time series example](https://github.com/elephaint/pgbm/tree/main/paper/experiments/02_hierarchical_time_series) from our paper or the [Covid-19 example](https://github.com/elephaint/pgbm/blob/main/examples/example10_covidhospitaladmissions.py). If a model has been loaded before `.train` is invoked, the model will continue training the loaded model. This allows for continual training and easy checkpointing. 

# GPU training #
Only applicable for the PyTorch backend. For training on GPU, it is required to set the following hyperparameters:
```
params['device'] = 'gpu'
```
When training on GPU, PGBM will select the GPU at the first index (0) by default and return the results at that device. This can be changed to e.g. the GPU at index 1 by setting the following hyperparameter:
```
params['gpu_device_id'] = 1
```

# Distributed CPU and GPU training #
Only applicable for the PyTorch backend. To facilitate distributed training, PGBM leverages the `torch.distributed` backend. In order to train in a distributed manner, we provide examples in the examples folder.

## Distributed training arguments ##
The following arguments can be set on the command line when using our template for distributed training:
* `-n`, `--nodes`, `default=1`: number of nodes used in the distributed training setting.
* `-p`, `--processes`, `default=1`: number of processes per node. For multi-GPU training, this should be equal to the number of GPUs per node.
* `-nr`, `--nr`, `default=0`: rank of the node within all nodes.
* `-b`, `--backend`, `default='gloo'`: backend for distributed training. Valid options are, dependent on your OS, PyTorch installation and distributed setting: `gloo`, `nccl`, `mpi`. 
* `-d`, `--device`, `default='cpu'`: device for training. Valid options: `cpu`, `gpu`.
* `--MASTER_ADDR`, `default='127.0.0.1'`: IP address of master process for distributed training.
* `--MASTER_PORT`, `default='29500'`: Port of node of master process for distributed training.

## Notes ##
* For more details on using distributed training with PyTorch, see [here](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
* Each process will have its own copy of the final PGBM model. Hence, to save a model, simply invoke a save method in one of the processes (i.e. `model.save(filename)`) or use checkpointing by setting `checkpoint=True` in the parameters dict.
* For multi-gpu training it is assumed that each node has the same amount of GPUs. 
* It should be possible to train with multiple GPUs of different types or generations, as long as it is a device with CUDA compute ability 6.x or up.

## Limitations ##
* It is possible to use autodifferentiation for custom loss functions during distributed training by setting `derivatives=approx`, however the gradient and hessian information is not (yet) shared across processes. This is not necessary for standard loss functions where the gradient information of example A depends only on the predicted value of example A, but for more complex loss functions this might be an issue (for example in the case of hierarchical time series forecasting). We hope to address this in future releases. 