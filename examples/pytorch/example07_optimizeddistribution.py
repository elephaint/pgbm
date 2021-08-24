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
#%% Import packages
import torch
from pgbm import PGBM
from sklearn.model_selection import train_test_split
from datasets import get_dataset, get_fold
#%% Objective
def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss

#%% Generic Parameters
# PGBM specific
method = 'pgbm'
params = {'min_split_gain':0,
      'min_data_in_leaf':2,
      'max_leaves':8,
      'max_bin':64,
      'learning_rate':0.1,
      'n_estimators':100,
      'verbose':2,
      'early_stopping_rounds':100,
      'feature_fraction':1,
      'bagging_fraction':1,
      'seed':1,
      'lambda':1,
      'device':'gpu',
      'gpu_device_id':0,
      'derivatives':'exact',
      'distribution':'normal'}
n_forecasts = 1000
#%% Loop
# datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht']
dataset = 'boston'
# Get data
data = get_dataset(dataset)
X_train, X_test, y_train, y_test = get_fold(dataset, data, 0)
X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
# Build datasets
train_data = (X_train, y_train)
train_val_data = (X_train_val, y_train_val)
valid_data = (X_val, y_val)
# Set base number of estimators
base_estimators = 2000
params['n_estimators'] = base_estimators
# Train to retrieve best iteration
model = PGBM()
model.train(train_val_data, objective=objective, metric=rmseloss_metric, valid_set=valid_data, params=params)
# Find best tree correlation & distribution on validation set
best_distribution, best_tree_correlation = model.optimize_distribution(X_val, y_val)
#%% Train model on full dataset and evaluate base case + optimal choice of distribution and correlation hyperparameter
# Set iterations to best iteration from validation set
params['n_estimators'] = model.best_iteration
# Retrain on full set   
model = PGBM()
model.train(train_data, objective=objective, metric=rmseloss_metric, params=params)
#% Probabilistic predictions base case
model.params['distribution'] = 'normal'
base_case_tree_correlation = model.params['tree_correlation']
yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
# Scoring
crps_old = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()
# Optimal case
model.params['distribution'] = best_distribution
model.params['tree_correlation'] = best_tree_correlation
yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
# Scoring
crps_new = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()  
# Print scores
print(f"Base case CRPS {crps_old:.2f}, distribution = normal, tree_correlation = {base_case_tree_correlation}")
print(f"Optimal CRPS {crps_new:.2f}, distribution = {model.params['distribution']}, tree_correlation = {model.params['tree_correlation']}")