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
import pandas as pd
import numpy as np
from datasets import get_dataset, get_fold
#%% Objective
def objective(yhat, y, levels=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    
    return gradient, hessian

def rmseloss_metric(yhat, y, levels=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss

#%% Generic Parameters
# PGBM specific
method = 'pgbm'
params = {'min_split_gain':0,
      'min_data_in_leaf':1,
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
      'tree_correlation':0.03,
      'device':'gpu',
      'output_device':'gpu',
      'gpu_device_ids':(0,),
      'derivatives':'exact',
      'distribution':'normal'}
n_forecasts = 1000
#%% Loop
# datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht']
dataset = 'boston'
df_val = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rho','distribution','crps_validation'])
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
tree_correlations = np.arange(10) * 0.01
distributions = ['normal','studentt','laplace','logistic','lognormal', 'gumbel', 'weibull', 'poisson', 'negativebinomial']
crps_pgbm = np.zeros((len(tree_correlations), len(distributions)))
for i, tree_correlation in enumerate(tree_correlations):
    for j, distribution in enumerate(distributions):
        print(f'Correlation {i+1} / distribution {j+1}')
        model.params['tree_correlation'] = tree_correlation
        model.params['distribution'] = distribution
        yhat_dist_pgbm = model.predict_dist(X_val, n_forecasts=n_forecasts)
        crps_pgbm[i, j] = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_val).mean()
        df_val = df_val.append({'method':method, 'dataset':dataset, 'fold':0, 'device':params['device'], 'validation_estimators': base_estimators, 'test_estimators':params['n_estimators'], 'rho': tree_correlation, 'distribution': distribution, 'crps_validation': crps_pgbm[i, j]}, ignore_index=True)   
#%% Train model on full dataset and evaluate base case + optimal choice of distribution and correlation hyperparameter
# Set iterations to best iteration from validation set
params['n_estimators'] = model.best_iteration + 1
# Retrain on full set   
model = PGBM()
model.train(train_data, objective=objective, metric=rmseloss_metric, params=params)
#% Probabilistic predictions base case
model.params['tree_correlation'] = 0.03
model.params['distribution'] = 'normal'
yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
# Scoring
crps_old = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()
# Optimal case
df_val_opt = df_val.loc[df_val.groupby(['dataset'])['crps_validation'].idxmin()]
model.params['tree_correlation'] = df_val_opt[df_val_opt.dataset == dataset]['rho'].item()
model.params['distribution'] = df_val_opt[df_val_opt.dataset == dataset]['distribution'].item()
yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
# Scoring
crps_new = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()  
# Print scores
print(f"Base case CRPS {crps_old:.2f}, distribution = normal, tree_correlation = 0.03")
print(f"Optimal CRPS {crps_new:.2f}, distribution = {model.params['distribution']}, tree_correlation = {model.params['tree_correlation']}")