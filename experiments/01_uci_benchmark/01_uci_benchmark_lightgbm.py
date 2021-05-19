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
import numpy as np
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import get_dataset, get_fold
#%% Objective
def rmseloss_metric(yhat, y):
    loss = np.sqrt(np.mean((yhat - y)**2))
    return loss    
#%% Generic Parameters
params = {'min_split_gain':0,
      'min_data_in_leaf':1,
      'max_depth':-1,
      'max_bin':64,
      'learning_rate':0.1,
      'n_estimators':2000,
      'verbose':2,
      'feature_fraction':1,
      'bagging_fraction':1,
      'seed':1}
#%% LightGBM specific
method = 'lightgbm'
params['lambda'] = 1
params['device'] = 'cpu'
params['objective'] = 'regression'
params['metric'] = 'rmse'
params['num_leaves'] = 8
params['bagging_freq'] = 1
params['min_data_in_bin'] = 1
#%% Loop
datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht', 'higgs']
base_estimators = 2000
df = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rmse_test','crps_test','validation_time'])
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
        train_data = lgb.Dataset(X_train, y_train)
        train_val_data = lgb.Dataset(X_train_val, y_train_val)
        valid_data = lgb.Dataset(X_val, y_val)
        test_data = lgb.Dataset(X_test, y_test)
        params['n_estimators'] = base_estimators
        # Train to retrieve best iteration
        print('Validating...')        
        start = time.perf_counter()    
        model = lgb.train(params, train_val_data, valid_sets=valid_data, early_stopping_rounds=2000)
        end = time.perf_counter()
        validation_time = end - start
        print(f'Fold time: {validation_time:.2f}s')
        # Set iterations to best iteration
        params['n_estimators'] = model.best_iteration + 1
        # Retrain on full set   
        print('Training...')
        model = lgb.train(params, train_data)
        #% Predictions
        print('Prediction...')
        yhat_point = model.predict(X_test)
        # Scoring
        rmse = rmseloss_metric(yhat_point, y_test)
        crps = 0
        # Save data
        df = df.append({'method':method, 'dataset':dataset, 'fold':fold, 'device':params['device'], 'validation_estimators': base_estimators, 'test_estimators':params['n_estimators'], 'rmse_test': rmse, 'crps_test': crps, 'validation_time':validation_time}, ignore_index=True)
#%% Save
filename = f"{method}_{params['device']}.csv"
df.to_csv(f'experiments/01_uci_benchmark/{filename}')