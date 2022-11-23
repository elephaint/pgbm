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
#%% Import packages
import time
from pgbm.sklearn import HistGradientBoostingRegressor, crps_ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from ngboost import NGBRegressor
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Parameters
params = {'min_samples_leaf':2,
      'max_leaf_nodes':8,
      'max_bins':64,
      'learning_rate':0.1,
      'max_iter':2000,
      'verbose':1,
      'early_stopping':True,
      'n_iter_no_change':2000,
      'validation_fraction':0.1,
      'random_state':1,
      'l2_regularization':1}
#%% Train pgbm vs NGBoost
n_splits = 2
n_forecasts = 1000
base_estimators = 2000
rmse, crps = np.zeros((n_splits, 2)), np.zeros((n_splits, 2))
#%% Loop
for i in range(n_splits):
    start = time.perf_counter()
    print(f'Fold {i+1}/{n_splits}')
    # Split for model validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    # Train to retrieve best iteration
    print('PGBM Validating on partial dataset...')
    params['max_iter'] = base_estimators
    params['early_stopping'] = True
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    # Set iterations to best iteration
    params['max_iter'] = np.argmax(model.validation_score_)
    params['early_stopping'] = False
    # Retrain on full set   
    print('PGBM Training on full dataset...')
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    #% Predictions
    print('PGBM Prediction...')
    yhat_point_pgbm, yhat_std = model.predict(X_test, return_std=True)
    yhat_dist_pgbm = model.sample(yhat_point_pgbm, yhat_std, n_estimates=100, random_state=i)
    # Scoring
    rmse[i, 0] = np.sqrt(mean_squared_error(yhat_point_pgbm, y_test))
    crps[i, 0] = crps_ensemble(yhat_dist_pgbm, y_test)         
    # NGB
    print('NGB Validating on partial dataset...')
    start = time.perf_counter()
    ngb = NGBRegressor(n_estimators=base_estimators)
    ngb.fit(X_train_val, y_train_val, X_val, y_val, early_stopping_rounds=2000)
    end = time.perf_counter()
    print(f'Fold time: {end - start:.2f}s')
    best_iter = ngb.best_val_loss_itr + 1
    ngb = NGBRegressor(n_estimators=best_iter)    
    print('NGB Training on full dataset...')
    ngb.fit(X_train, y_train)
    print('NGB Prediction...')    
    yhat_point_ngb = ngb.predict(X_test)
    ngb_dist = ngb.pred_dist(X_test)
    yhat_dist_ngb = ngb_dist.sample(n_forecasts)
    # Scoring NGB
    rmse[i, 1] = np.sqrt(mean_squared_error(yhat_point_ngb, y_test))
    crps[i, 1] = crps_ensemble(np.asfortranarray(yhat_dist_ngb), y_test)        
    # Print scores current fold
    print(f'RMSE Fold {i+1}, PGBM: {rmse[i, 0]:.2f}, NGB: {rmse[i, 1]:.2f}')
    print(f'CRPS Fold {i+1}, PGBM: {crps[i, 0]:.2f}, NGB: {crps[i, 1]:.2f}')
      
# Print final scores
print(f'RMSE PGBM: {rmse[:, 0].mean():.2f}+-{rmse[:, 0].std():.2f}, NGB: {rmse[:, 1].mean():.2f}+-{rmse[:, 1].std():.2f}')
print(f'CRPS PGBM: {crps[:, 0].mean():.2f}+-{crps[:, 0].std():.2f}, NGB: {crps[:, 1].mean():.2f}+-{crps[:, 1].std():.2f}')
#%% Plot all sample
plt.plot(y_test, 'o', label='Actual')
plt.plot(yhat_point_pgbm, 'ko', label='Point prediction PGBM')
plt.plot(yhat_point_ngb, 'ro', label='Point prediction NGBoost')
plt.plot(yhat_dist_pgbm.max(axis=0), 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.min(axis=0), 'k--', label='Min bound PGBM')
plt.plot(yhat_dist_ngb.max(axis=0), 'r--', label='Max bound NGBoost')
plt.plot(yhat_dist_ngb.min(axis=0), 'r--', label='Min bound NGBoost')
plt.legend()