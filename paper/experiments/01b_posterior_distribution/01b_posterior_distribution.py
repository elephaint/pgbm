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
import torch
import time
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
n_forecasts = 1000
#%% Loop
datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht']
# datasets = ['energy']
base_estimators = 2000
df_val = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rho','distribution','crps_validation'])
for i, dataset in enumerate(datasets):
    if dataset == 'msd':
        params['bagging_fraction'] = 0.1
        params['device'] = 'gpu'
        device = torch.device(0)
    else:
        params['bagging_fraction'] = 1
        params['device'] = 'cpu'
        device = torch.device('cpu')
    # Get data
    data = get_dataset(dataset)
    X_train, X_test, y_train, y_test = get_fold(dataset, data, 0)
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    # Build datasets
    train_data = (X_train, y_train)
    train_val_data = (X_train_val, y_train_val)
    valid_data = (X_val, y_val)
    params['n_estimators'] = base_estimators
    # Train to retrieve best iteration
    print('Validating...')
    model = PGBM()
    start = time.perf_counter()    
    model.train(train_val_data, objective=objective, metric=rmseloss_metric, valid_set=valid_data, params=params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    validation_time = end - start
    print(f'Fold time: {validation_time:.2f}s')
    # Find best tree correlation on validation set
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
#%% Save file
filename_val = "distribution_variation_validation.csv"
df_val.to_csv(f'pgbm/experiments/01b_posterior_distribution/{filename_val}')    
#%% Base case + optimal
df_base = pd.read_csv('pgbm/experiments/01_uci_benchmark/pgbm_gpu.csv', index_col=0)
fold = 0
df_base = df_base[df_base.fold == fold]
df_test = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rho','distribution','rmse_test','crps_test','validation_time','case'])
df_val = pd.read_csv(f'pgbm/experiments/01b_posterior_distribution/{filename_val}', index_col=0)  
df_val = df_val.loc[df_val.groupby(['dataset'])['crps_validation'].idxmin()]
for i, dataset in enumerate(datasets):
    if dataset == 'msd':
        params['bagging_fraction'] = 0.1
        params['device'] = 'gpu'
        device = torch.device(0)
    else:
        params['bagging_fraction'] = 1
        params['device'] = 'cpu'
        device = torch.device('cpu')
    # Get data
    data = get_dataset(dataset)
    X_train, X_test, y_train, y_test = get_fold(dataset, data, 0)
    train_data = (X_train, y_train)
    # Set iterations to best iteration
    params['n_estimators'] = int(df_base[df_base.dataset == dataset]['test_estimators'])
    # Retrain on full set   
    print('Training...')
    model = PGBM()
    model.train(train_data, objective=objective, metric=rmseloss_metric, params=params)
    #% Predictions base case
    print('Prediction...')
    yhat_point_pgbm = model.predict(X_test)
    model.params['tree_correlation'] = np.log10(len(X_train)) / 100
    model.params['distribution'] = 'normal'
    yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
    # Scoring
    rmse_pgbm_new = model.metric(yhat_point_pgbm.cpu(), y_test).numpy()
    crps_pgbm_new = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()
    # Save data
    df_test = df_test.append({'method':method, 'dataset':dataset, 'fold':fold, 'device':params['device'], 'validation_estimators': base_estimators, 'test_estimators':params['n_estimators'], 'rmse_test': rmse_pgbm_new, 'crps_test': crps_pgbm_new, 'validation_time':validation_time, 'rho':0.03 ,'distribution':'normal','case':'base'}, ignore_index=True)
    # Optimal case
    row, col = np.unravel_index(crps_pgbm.argmin(), crps_pgbm.shape)
    model.params['tree_correlation'] = df_val[df_val.dataset == dataset]['rho'].item()
    model.params['distribution'] = df_val[df_val.dataset == dataset]['distribution'].item()
    yhat_dist_pgbm = model.predict_dist(X_test, n_forecasts=n_forecasts)
    # Scoring
    rmse_pgbm_new = model.metric(yhat_point_pgbm.cpu(), y_test).numpy()
    crps_pgbm_new = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean() 
    # Save data
    df_test = df_test.append({'method':method, 'dataset':dataset, 'fold':fold, 'device':params['device'], 'validation_estimators': base_estimators, 'test_estimators':params['n_estimators'], 'rmse_test': rmse_pgbm_new, 'crps_test': crps_pgbm_new.numpy().item(), 'validation_time':validation_time, 'rho':model.params['tree_correlation'] ,'distribution':model.params['distribution'],'case':'optimal'}, ignore_index=True)

#%% Save data
filename_test = "distribution_variation_test.csv"
df_test.to_csv(f'pgbm/experiments/01b_posterior_distribution/{filename_test}')        
