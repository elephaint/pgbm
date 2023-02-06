"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

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

#%% Load packages
from pgbm.sklearn import HistGradientBoostingRegressor, crps_ensemble
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Parameters
params = {'min_samples_leaf':2,
      'max_leaf_nodes':8,
      'max_bins':64,
      'learning_rate':0.1,
      'max_iter':2000,
      'verbose':2,
      'early_stopping':True,
      'n_iter_no_change':100,
      'validation_fraction':0.1,
      'random_state':1,
      'l2_regularization':1}

n_forecasts = 1000
n_splits = 2
base_estimators = 2000
#%% Validation loop
rmse, crps = np.zeros(n_splits), np.zeros(n_splits)
for i in range(n_splits):
    print(f'Fold {i+1}/{n_splits}')
    # Split for model validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    # Train to retrieve best iteration
    print('PGBM Validating on partial dataset...')
    params['max_iter'] = base_estimators
    params['early_stopping'] = True
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    # Set iterations to best iteration
    params['max_iter'] = model.n_iter_
    params['early_stopping'] = False
    # Retrain on full set   
    print('PGBM Training on full dataset...')
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    #% Predictions
    print('PGBM Prediction...')
    yhat_point, yhat_std = model.predict(X_test, return_std=True)
    yhat_dist = model.sample(yhat_point, yhat_std, n_estimates=100, random_state=i)
    # Scoring
    rmse[i] = np.sqrt(mean_squared_error(yhat_point, y_test))
    crps[i] = crps_ensemble(yhat_dist, y_test)         
    # Print scores current fold
    print(f'RMSE Fold {i+1}, {rmse[i]:.2f}')
    print(f'CRPS Fold {i+1}, {crps[i]:.2f}')
      
# Print final scores
print(f'RMSE {rmse.mean():.2f}+-{rmse.std():.2f}')
print(f'CRPS {crps.mean():.2f}+-{crps.std():.2f}')
#%% Plot all samples
plt.plot(y_test, 'o', label='Actual')
plt.plot(yhat_point, 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist.max(axis=0), 'k--', label='Max bound PGBM')
plt.plot(yhat_dist.min(axis=0), 'k--', label='Min bound PGBM')
plt.legend()
