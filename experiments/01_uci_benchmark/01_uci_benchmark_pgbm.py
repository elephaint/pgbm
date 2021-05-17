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
import properscoring as ps
import pandas as pd
import numpy as np
from pgbm.datasets import get_dataset, get_fold
#%% Objective
def objective(yhat, y):
    gradient = (yhat - y)
    hessian = 1 + yhat*0.
    
    return gradient, hessian

def rmseloss_metric(yhat, y):
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
      'device':'cpu',
      'output_device':'gpu',
      'gpu_device_ids':(0,),
      'derivatives':'exact',
      'distribution':'normal'}
n_samples = 1000
#%% Loop
datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht','higgs']
base_estimators = 2000
df = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rmse_test','crps_test','validation_time'])
torchdata = lambda x : torch.from_numpy(x).float()
for i, dataset in enumerate(datasets):
    if dataset == 'msd' or dataset == 'higgs':
        params['bagging_fraction'] = 0.1
        n_folds = 1
    else:
        params['bagging_fraction'] = 1
        n_folds = 20
    data = get_dataset(dataset)
    for fold in range(n_folds):
        print(f'{dataset}: fold {fold + 1}/{n_folds}')
        # Get data
        X_train, X_test, y_train, y_test = get_fold(dataset, data, fold)
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=fold)
        # Build torchdata datasets
        train_data = (torchdata(X_train), torchdata(y_train))
        train_val_data = (torchdata(X_train_val), torchdata(y_train_val))
        valid_data = (torchdata(X_val), torchdata(y_val))
        test_data = (torchdata(X_test), torchdata(y_test))
        params['n_estimators'] = base_estimators
        # Train to retrieve best iteration
        print('Validating...')
        model = pgbm.PGBM(params)
        start = time.perf_counter()    
        model.train(train_val_data, objective=objective, metric=rmseloss_metric, valid_set=valid_data)
        torch.cuda.synchronize()
        end = time.perf_counter()
        validation_time = end - start
        print(f'Fold time: {validation_time:.2f}s')
        # Set iterations to best iteration
        params['n_estimators'] = model.best_iteration + 1
        # Retrain on full set   
        print('Training...')
        model = pgbm.PGBM(params)
        model.train(train_data, objective=objective, metric=rmseloss_metric)
        #% Predictions
        print('Prediction...')
        yhat_point = model.predict(test_data[0])
        model.params['tree_correlation'] = np.log10(len(X_train)) / 100
        yhat_dist = model.predict_dist(test_data[0], n_samples=n_samples)
        # Scoring
        rmse = rmseloss_metric(yhat_point.cpu(), test_data[1]).numpy()
        crps = ps.crps_ensemble(y_test, yhat_dist.cpu().T).mean()
        # Save data
        df = df.append({'method':method, 'dataset':dataset, 'fold':fold, 'device':params['device'], 'validation_estimators': base_estimators, 'test_estimators':params['n_estimators'], 'rmse_test': rmse, 'crps_test': crps, 'validation_time':validation_time}, ignore_index=True)
#%% Save
filename = f"{method}_{params['device']}.csv"
df.to_csv(f'pgbm/experiments/01_uci_benchmark/{filename}')