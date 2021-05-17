# -*- coding: utf-8 -*-
"""
"""
#%%
import torch
import time
import pgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
import properscoring as ps
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
rmse, crps = np.zeros(n_splits), np.zeros(n_splits)
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
    print('PGBM Training on full dataset...')
    model = pgbm.PGBM(params)
    model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
    #% Predictions
    print('PGBM Prediction...')
    yhat_point_pgbm = model.predict(test_data[0])
    yhat_dist_pgbm = model.predict_dist(test_data[0], n_samples=n_samples)
    # Scoring
    rmse[i] = rmseloss_metric(yhat_point_pgbm.cpu(), y_test).numpy()
    crps[i] = ps.crps_ensemble(y_test, yhat_dist_pgbm.cpu().T).mean()
    # Print scores current fold
    print(f'RMSE Fold {i+1}, PGBM: {rmse[i]:.2f}')
    print(f'CRPS Fold {i+1}, PGBM: {crps[i]:.2f}')
      
# Print final scores
print(f'RMSE PGBM: {rmse.mean():.2f}+-{rmse.std():.2f}')
print(f'CRPS PGBM: {crps.mean():.2f}+-{crps.std():.2f}')
#%% Plot all sample
plt.plot(test_data[1], 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.legend()