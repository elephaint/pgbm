# Parameters #

## Training parameters (Scikit-learn backend) ##
`HistGradientBoostingRegressor` has the same hyperparameters as [Scikit-learn's normal version](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), plus:
* `distribution`, default=`normal`. Choice of output distribution for probabilistic predictions. Choices are `normal`, `studentt`, `laplace`, `logistic`, `lognormal`, `gamma`, `gumbel`, `poisson`, `negativebinomial`.
* `tree_correlation`, default=`log_10(n_samples_train) / 100`. Tree correlation hyperparameter. This controls the amount of correlation we assume to exist between each subsequent tree in the ensemble. 
* `studentt_degrees_of_freedom`, default=`3`. Degrees of freedom, only used for Student-t distribution when sampling probabilistic predictions using the `sample` method.
* `with_variance`, default=`True`. Whether to store the variances of the leafs during training. If set to `False`, `HistGradientBoostingRegressor` will perform identical to Scikit-learn's normal version. Keep this to `True` if you want to be able to create probabilistic estimates.

## Setting parameters ##
Let's recall the example from our [Quick Start section](./quick_start.md):
```
from pgbm.sklearn import HistGradientBoostingRegressor, crps_ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
model = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)  
yhat_point, yhat_point_std = model.predict(X_test, return_std=True)
yhat_dist = model.sample(yhat_point, yhat_point_std, n_estimates=1000, random_state=0)
```
To assess our forecast, we compute an error metric. For the point forecast, we use Root Mean Squared Error [(RMSE)](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and we use Continuously Ranked Probability Score [(CRPS)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) for the probabilistic forecast. 
```
rmse = np.sqrt(mean_squared_error(yhat_point, y_test))
crps = crps_ensemble(yhat_dist, y_test)    
```
which should result in an RMSE of `0.456` and a CRPS of `0.227`. 

Suppose we would like to improve on this result, for example by increasing the number of estimators from the default of `100` to `200`:
```
model = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X_train, y_train)  
```
This should result in an improved RMSE of `0.440` and a CRPS of `0.220` on the test set.

This demonstrates how to set one of the parameters to make an improvement over the baseline result. 

## Parameter tuning ##
We can use external hyperparameter optimization tools such as [Optuna](https://optuna.org/) as well as scikit-learn's [built-in hyperparameter tuning methods](https://scikit-learn.org/stable/modules/grid_search.html#grid-search). For the latter, we provide an example [here](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example07_hyperparamoptimization.py) and [here](https://github.com/elephaint/pgbm/blob/main/examples/sklearn/example14_probregression.py).


