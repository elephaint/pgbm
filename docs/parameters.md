# Hyperparameters #

## Training parameters ##
PGBM employs the following set of hyperparameters (listed in alphabetical order):
* `bagging_fraction`, default=`1`. Fraction of samples to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of samples to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `checkpoint`, default=`False`. Boolean to save a model checkpoint after each iteration to the current working directory. 
* `derivatives`, default=`exact`. If a loss function with an analytical gradient and hessian is provided, use `exact`. If a loss function with a scalar, differentiable loss is provided, use `approx` to let PyTorch use auto-differentiation to calculate the gradient and (approximate) hessian. Not applicable for Numba backend.
* `device`, default=`cpu`. Traininig device. Choices are `cpu` or `gpu`. Not applicable for Numba backend.
* `distribution`, default=`normal`. Choice of output distribution for probabilistic predictions. Choices are `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `weibull`, `poisson`, `negativebinomial`. For the Numba backend, only `normal`, `studentt`, `laplace`, `logistic`, `gamma`, `gumbel`, `poisson` are supported. Note that the `studentt` distribution has a constant degree-of-freedom of `3`.
* `early_stopping_rounds`, default = `100`. The number of iterations after which the training stops should the validation metric not improve. Only applicable in case a validation set is used.
* `feature_fraction`, default=`1`. Fraction of features to use when building a tree. Set to a value between `0` and `1` to randomly select a portion of features to construct each new tree. A lower fraction speeds up training (and can be used to deal with out-of-memory issues when training on GPU) and may reduce overfit.
* `gpu_device_id`, default=`0`. Integer with the index of the GPU used for training. Change this when you'd like to train on a different GPU within your node. For multi-gpu and multinode training, see [here](./distributed_learning.md). Not applicable for Numba backend.
* `lambda`, default=`1`, constraints`>0`. Regularization parameter. 
* `learning_rate`, default=`0.1`. The learning rate of the algorithm; the amount of each new tree prediction that should be added to the ensemble.
* `max_bin`, default=`256`, constraint`<32,767`. The maximum number of bins used to bin continuous features. Increasing this value can improve prediction performance, at the cost of training speed and potential overfit. 
* `max_leaves`, default=`32`. The maximum number of leaves per tree. Increase this value to create more complicated trees, and reduce the value to create simpler trees (reduce overfitting).
* `min_data_in_leaf`, default= `3`, constraint`>= 3`. The minimum number of samples in a leaf of a tree. Increase this value to reduce overfit.
* `min_split_gain`, default = `0.0`. The minimum gain for a node to split when building a tree.
* `monotone_constraints`, default = `zeros of shape [n_features]`. This allows to provide monotonicity constraints per feature. `1` means increasing, `0` means no constraint, `-1` means decreasing. All features need to be specified when using this parameters, for example `monotone_constraints=[1, 0, -1]` for a positive, non-constraint and negative constraint for respectively feature 1, 2 and 3. There should be limited effect on training speed. To improve accuracy, you can try to increase `monotone_iterations` (see hereafter), but this comes at the expense of slower training. 
* `monotone_iterations`, default=`1`. The number of alternative splits that will be considered if a monotone constraint is violated by the current split proposal. Increase this to improve accuracy at the expense of training speed.
* `n_estimators`, default=`100`. The number of trees to create. Typically setting this value higher may improve performance, at the expense of training speed and potential for overfit. Use in conjunction with `learning rate` and `max_leaves`; more trees generally requires a lower `learning_rate` and/or a lower `max_leaves`.
* `seed`, default=`2147483647`. Random seed to use for `feature_fraction` and `bagging_fraction` (latter only for Numba backend - for speed considerations the Torch backend `bagging_fraction` determination is not yet deterministic).
* `split_parallel`, default=`feature`. Choose from `feature` or `sample`. This parameter determines whether to parallelize the split decision computation across the sample dimension or across the feature dimension. Typically, for smaller datasets with few features `feature` is the fastest, whereas for larger datasets and/or datasets with many (e.g. > 50) features, `sample` will provide better results. Only applicable when using the Numba backend.
* `tree_correlation`, default=`log_10(n_samples_train) / 100`. Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble. 
* `verbose`, default=`2`. Flag to output metric results for each iteration. Set to `1` to supress output.

__GPU training__
Only applicable for the PyTorch backend. For training on GPU, it is required to set the following hyperparameters:
```
params['device'] = 'gpu'
```
When training on GPU, PGBM will select the GPU at the first index (0) by default and return the results at that device. This can be changed to e.g. the GPU at index 1 by setting the following hyperparameter:
```
params['gpu_device_id'] = 1
```

## Setting parameters ##
Let's recall the example from our [Quick Start section](./quick_start.md):
```
from pgbm import PGBMRegressor # If you want to use the Torch backend
# from pgbm_nb import PGBMRegressor # If you want to use the Numba backend
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
model = PGBMRegressor().fit(X_train, y_train)  
yhat_point = model.predict(X_test) # Point predictions
yhat_dist = model.predict_dist(X_test) # Probabilistic predictions
```
To assess our forecast, we compute an error metric. For the point forecast, we use Root Mean Squared Error [(RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and we use Continuously Ranked Probability Score [(CRPS)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) for the probabilistic forecast. Both error metrics are built-in and can be called as follows:
```
rmse = model.rmseloss_metric(yhat_point, y_test)
crps = model.crps_ensemble(yhat_dist, y_test).mean()  
```
which should result in an RMSE of `0.474` and a CRPS of `0.241`. 

Suppose we would like to improve on this result, for example by increasing the number of estimators from the default of `100` to `200`:
```
model = PGBMRegressor(n_estimators=200).fit(X_train, y_train)  
```
This should result in an improved RMSE of `0.446` and a CRPS of `0.228` on the test set.

This demonstrates how to set one of the parameters to make an improvement over the baseline result. 

## Parameter tuning ##
We can use external hyperparameter optimization tools such as [Optuna](https://optuna.org/) to perform the parameter tuning in an automatic fashion. In the following code snippet we perform hyperparameter optimization for two hyperparameters. More specifically:
* We are interested in the best settings for `bagging_fraction` and `feature_fraction` (see above for explanation).
* We run 100 trials where each trial evaluates a different set of values for these hyperparameters.
* We use sklearn's `cross_val_score` using 5 folds (`cv=5`) with scoring `neg_root_mean_squared_error`, as the latter is the same metric we were previously interested in. Note that because this scoring function uses the negative error, we need to set the direction in our Optuna study to `maximize`. 
* We use GPU-training, and because training a PGBM model with this dataset does not saturate our GPU (RTX 2080Ti) both in CUDA core load as well as memory usage, we set `n_jobs=5` to simultaneously run all 5 folds per trial on the same GPU, which completely saturates our GPU in CUDA core load. This significantly increases speed of the hyperparameter optimization and increasing the number of jobs should be considered if your GPU or CPU is not yet saturated running only one fold.
* Within each fold, we train each model for 200 iterations.

```
import optuna
from pgbm import PGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import fetch_california_housing
import numpy as np


class Objective(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __call__(self, trial):
        # 1. Suggest values of the hyperparameters
        params = {'n_estimators': 200,
                  'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
                  'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
                  'device':'gpu'
                  }
        model = PGBMRegressor()
        model.set_params(**params)
        # 2. Fit model and score
        score = np.mean(cross_val_score(model, self.X, self.y, cv=5, n_jobs=5, scoring='neg_root_mean_squared_error'))

        return score

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
objective = Objective(X_train, y_train)
study.optimize(objective, n_trials=100)
print(study.best_trial)
```
After the optimization, we can use the best parameters to fit our final model.
```
# 4. Set best parameters to a model and fit
model = PGBMRegressor()
model.set_params(**study.best_params)
model.set_params(n_estimators=200, device='gpu')
model.fit(X_train, y_train)
```
Now, we can evaluate the fitted model on our test set to see if we improved over the baseline:
```
# 5. Evaluate on test set
yhat_point = model.predict(X_test) # Point predictions
yhat_dist = model.predict_dist(X_test) # Probabilistic predictions
rmse = model.rmseloss_metric(yhat_point, y_test)
crps = model.crps_ensemble(yhat_dist, y_test).mean() 
```
This results in an RMSE of `0.447` and a CRPS of `0.228`, hence we didn't improve over our baseline result but feel free to improve over our result!



