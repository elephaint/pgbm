"""
   Copyright (c) 2021 Olivier Sprangers 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   https://github.com/elephaint/pgbm/blob/main/LICENSE

"""
#%% Import packagess
import numpy as np
from pgbm.sklearn import (
    HistGradientBoostingRegressor, 
    crps_ensemble, 
    make_probabilistic_scorer
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from datasets import get_dataset, get_fold
#%% Generic Parameters
# PGBM specific
params = {
      'max_iter':2000,
      'early_stopping':True,
      'random_state':1,
      'n_iter_no_change':100}
n_forecasts = 1000
#%% Dataset
# datasets = ['housing', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht']
dataset = 'concrete'
# Get data
data = get_dataset(dataset)
X_train, X_test, y_train, y_test = get_fold(dataset, data, 0)
#%% Train base model
model = HistGradientBoostingRegressor(**params)
model.fit(X_train, y_train)
# Predict on test set
yhat_point, yhat_std = model.predict(X_test, return_std=True)
yhat_dist = model.sample(yhat_point, yhat_std, n_estimates=100, random_state=1)
# Base RMSE & CRPS on test set
rmse_base = np.sqrt(mean_squared_error(yhat_point, y_test))
crps_base = crps_ensemble(yhat_dist, y_test)         
#%% Optimize model using cross-validation
score_func = make_probabilistic_scorer(crps_ensemble, greater_is_better=False)
# Define the parameter grid.
param_grid = dict(
    min_samples_leaf=[1, 5, 10, 20],
    tree_correlation=np.linspace(-0.05, 0.05, 10),
    distribution=["normal", "laplace", "studentt", "logistic", "gumbel"],
)
# Fit with GridSearchCV
gbr = HistGradientBoostingRegressor(**params)
search = GridSearchCV(
    gbr,
    param_grid,
    scoring=score_func,
    verbose=1,
    cv=3,
    n_jobs=-1
).fit(X_train, y_train)
# Predict on test set
yhat_point, yhat_std = search.best_estimator_.predict(X_test, return_std=True)
yhat_dist = search.best_estimator_.sample(yhat_point, yhat_std, n_estimates=100, random_state=1)
# Optimized RMSE & CRPS on test set
rmse_optim = np.sqrt(mean_squared_error(yhat_point, y_test))
crps_optim = crps_ensemble(yhat_dist, y_test)         
#%% Print scores
print(f"Base case CRPS: {crps_base:.2f}, distr = {model.distribution}, tree_correlation = {model._tree_correlation:.3f}")
print(f"Optimal CRPS  : {crps_optim:.2f}, distr = {search.best_estimator_.distribution}, tree_correlation = {search.best_estimator_._tree_correlation:.3f}")
#%% Note that (in this case) RMSE was not impacted if we optimize for probabilistic performance!
print(f"Base case RMSE: {rmse_base:.2f}")
print(f"Optimal RMSE  : {rmse_optim:.2f}")
