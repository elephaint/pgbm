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
from pgbm_nb import PGBM
import numpy as np
import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from datasets import get_dataset, get_fold
#%% Objective for pgbm
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = np.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = np.sqrt(np.mean(np.square(yhat - y)))

    return loss
#%% Load data
#datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht','higgs']
dataset = 'protein'
data = get_dataset(dataset)
X_train, X_test, y_train, y_test = get_fold(dataset, data, random_state=0)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
#%% Parameters
base_estimators = 500
params = {'min_split_gain':0,
      'min_data_in_leaf':1,
      'max_leaves':32,
      'max_bin':256,
      'learning_rate':0.1,
      'n_estimators':base_estimators,
      'verbose':2,
      'early_stopping_rounds':base_estimators,
      'feature_fraction':1,
      'bagging_fraction':0.1,
      'seed':1,
      'lambda':1,
      'tree_correlation':0.03,
      'split_parallel':'feature',
      'distribution':'normal'}
#%% PGBM-Numba
# Tuples of datasets in tensors
train_val_data = (X_train_val, y_train_val)
valid_data = (X_val, y_val)
# Train 1 iteration to compile code
params['n_estimators'] = 1
model = PGBM()    
model.train(train_set=train_val_data, objective=mseloss_objective, metric=rmseloss_metric, valid_set=valid_data, params=params)
# Train to retrieve best iteration
params['n_estimators'] = base_estimators
start = time.perf_counter()
model = PGBM()    
model.train(train_set=train_val_data, objective=mseloss_objective, metric=rmseloss_metric, valid_set=valid_data, params=params)
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
