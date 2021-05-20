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
import torch
import time
import pgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import properscoring as ps
from ngboost import NGBRegressor
#%% Objective for pgbm
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = yhat*0. + 1

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = load_boston(return_X_y=True)
#%% Parameters
params = {'min_split_gain':0,
      'min_data_in_leaf':1,
      'max_leaves':8,
      'max_bin':64,
      'learning_rate':0.1,
      'n_estimators':2000,
      'verbose':2,
      'early_stopping_rounds':2000,
      'feature_fraction':1,
      'bagging_fraction':1,
      'seed':1,
      'lambda':1,
      'tree_correlation':0.03,
      'device':'cpu',
      'output_device':'gpu',
      'gpu_device_ids':(0,),
      'derivatives':'exact',
      'distribution':'normal'}
#%% Train pgbm vs NGBoost
n_splits = 1
n_samples = 1000
base_estimators = 200
rmse, crps = np.zeros((n_splits, 2)), np.zeros((n_splits, 2))
#%% Loop
torchdata = lambda x : torch.from_numpy(x).float()
for i in range(n_splits):
    start = time.perf_counter()
    print(f'Fold {i+1}/{n_splits}')
    # Split for model validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    # Build torchdata datasets
    train_data = (torchdata(X_train), torchdata(y_train))
    train_val_data = (torchdata(X_train_val), torchdata(y_train_val))
    valid_data = (torchdata(X_val), torchdata(y_val))
    test_data = (torchdata(X_test), torchdata(y_test))
    # Train to retrieve best iteration
    print('PGBM Validating on partial dataset...')
    params['n_estimators'] = base_estimators
    start = time.perf_counter()    
    model = pgbm.PGBM(params)
    model.train(train_val_data, objective=mseloss_objective, metric=rmseloss_metric, valid_set=valid_data)
    end = time.perf_counter()
    print(f'Fold time: {end - start:.2f}s')
    # Set iterations to best iteration
    params['n_estimators'] = model.best_iteration + 1
    # Retrain on full set   
    print(f'PGBM Training on full dataset...')
    model = pgbm.PGBM(params)
    model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
    #% Predictions
    print(f'PGBM Prediction...')
    yhat_point_pgbm = model.predict(test_data[0])
    yhat_dist_pgbm = model.predict_dist(test_data[0], n_samples=n_samples)
    # Scoring
    rmse[i, 0] = rmseloss_metric(yhat_point_pgbm.cpu(), y_test).numpy()
    crps[i, 0] = ps.crps_ensemble(y_test, yhat_dist_pgbm.cpu().T).mean()
    # NGB
    print(f'NGB Validating on partial dataset...')
    start = time.perf_counter()
    ngb = NGBRegressor(n_estimators=base_estimators)
    ngb.fit(X_train_val, y_train_val, X_val, y_val, early_stopping_rounds=2000)
    end = time.perf_counter()
    print(f'Fold time: {end - start:.2f}s')
    best_iter = ngb.best_val_loss_itr + 1
    ngb = NGBRegressor(n_estimators=best_iter)    
    print(f'NGB Training on full dataset...')
    ngb.fit(X_train, y_train)
    print(f'NGB Prediction...')    
    yhat_point_ngb = ngb.predict(X_test)
    ngb_dist = ngb.pred_dist(X_test)
    yhat_dist_ngb = ngb_dist.sample(n_samples)
    # Scoring NGB
    rmse[i, 1] = rmseloss_metric(torch.from_numpy(yhat_point_ngb), test_data[1]).numpy()
    crps[i, 1] = ps.crps_ensemble(test_data[1], yhat_dist_ngb.T).mean()        
    # Print scores current fold
    print(f'RMSE Fold {i+1}, PGBM: {rmse[i, 0]:.2f}, NGB: {rmse[i, 1]:.2f}')
    print(f'CRPS Fold {i+1}, PGBM: {crps[i, 0]:.2f}, NGB: {crps[i, 1]:.2f}')
      
# Print final scores
print(f'RMSE PGBM: {rmse[:, 0].mean():.2f}+-{rmse[:, 0].std():.2f}, NGB: {rmse[:, 1].mean():.2f}+-{rmse[:, 1].std():.2f}')
print(f'CRPS PGBM: {crps[:, 0].mean():.2f}+-{crps[:, 0].std():.2f}, NGB: {crps[:, 1].mean():.2f}+-{crps[:, 1].std():.2f}')
#%% Plot all sample
plt.plot(test_data[1], 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_point_ngb, 'ro', label='Point prediction NGBoost')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.plot(yhat_dist_ngb.max(axis=0), 'r--', label='Max bound NGBoost')
plt.plot(yhat_dist_ngb.min(axis=0), 'r--', label='Min bound NGBoost')
plt.legend()