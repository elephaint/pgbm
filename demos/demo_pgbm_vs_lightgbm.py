# -*- coding: utf-8 -*-
"""
"""
#%%
import torch
import time
import pgbm
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from pgbm.datasets import get_dataset, get_fold
#%% Objective for pgbm
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = yhat*0. + 1

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
#datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht','higgs']
dataset = 'higgs'
data = get_dataset(dataset)
X_train, X_test, y_train, y_test = get_fold(dataset, data, random_state=0)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#%% Parameters
params = {'min_split_gain':0,
      'min_data_in_leaf':1,
      'max_leaves':8,
      'max_bin':64,
      'learning_rate':0.1,
      'n_estimators':100,
      'verbose':2,
      'early_stopping_rounds':2000,
      'feature_fraction':1,
      'bagging_fraction':1,
      'seed':1,
      'lambda':1,
      'tree_correlation':0.03,
      'device':'gpu',
      'output_device':'gpu',
      'gpu_device_ids':(0,),
      'derivatives':'exact',
      'distribution':'normal'}
#%% PGBM
# Tuples of datasets in torch tensors
torchdata = lambda x : torch.from_numpy(x).float()
train_val_data = (torchdata(X_train_val), torchdata(y_train_val))
valid_data = (torchdata(X_val), torchdata(y_val))
# Train to retrieve best iteration
start = time.perf_counter()
model = pgbm.PGBM(params)    
model.train(train_set=train_val_data, objective=mseloss_objective, metric=rmseloss_metric, valid_set=valid_data)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Fold time: {end - start:.2f}s')
#%% LightGBM
# Additional parameters
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['bagging_freq'] = 1
params['min_data_in_bin'] = 1
params['bin_construct_sample_cnt'] = len(X_train_val)
params['device'] = 'cpu'
# Train LightGBM
train_val_data_lgb = lgb.Dataset(X_train_val, y_train_val)
valid_data_lgb = lgb.Dataset(X_val, y_val)
start = time.perf_counter()
lgb_model = lgb.train(params, train_val_data_lgb, valid_sets=valid_data_lgb)
end = time.perf_counter()
print(f'Fold time: {end - start:.2f}s')
