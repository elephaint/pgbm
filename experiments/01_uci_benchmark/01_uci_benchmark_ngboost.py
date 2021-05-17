# -*- coding: utf-8 -*-
"""
"""
import time
from sklearn.model_selection import train_test_split
from ngboost import NGBRegressor
import properscoring as ps
import pandas as pd
import numpy as np
from pgbm.datasets import get_dataset, get_fold
#%% specific
def rmseloss_metric(yhat, y):
    loss = np.sqrt(np.mean((yhat - y)**2))
    return loss    

method = 'ngboost'
n_samples = 1000
#%% Loop
datasets = ['boston', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht','higgs']
base_estimators = 2000
df = pd.DataFrame(columns=['method', 'dataset','fold','device','validation_estimators','test_estimators','rmse_test','crps_test','validation_time'])
for i, dataset in enumerate(datasets):
    if dataset == 'msd' or dataset == 'higgs':
        n_folds = 1
    else:
        n_folds = 20
    data = get_dataset(dataset)
    for fold in range(n_folds):
        print(f'{dataset}: fold {fold + 1}/{n_folds}')
        # Get data
        X_train, X_test, y_train, y_test = get_fold(dataset, data, fold)
        X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=fold)
        # Train to retrieve best iteration
        print('Validating...')
        if dataset == 'msd' or dataset == 'higgs':
            ngb = NGBRegressor(n_estimators=base_estimators, minibatch_frac=0.1, learning_rate=0.1)
        else:
            ngb = NGBRegressor(n_estimators=base_estimators)
        start = time.perf_counter()    
        ngb.fit(X_train_val, y_train_val, X_val, y_val, early_stopping_rounds=2000)
        end = time.perf_counter()
        validation_time = end - start
        print(f'Fold time: {validation_time:.2f}s')
        # Set iterations to best iteration
        best_iter = ngb.best_val_loss_itr + 1
        # Retrain on full set    
        print('Training...')
        if dataset == 'msd' or dataset == 'higgs':
            ngb = NGBRegressor(n_estimators=best_iter, minibatch_frac=0.1, learning_rate=0.1)
        else:
            ngb = NGBRegressor(n_estimators=best_iter)
        ngb.fit(X_train, y_train)
        #% Predictions
        print('Prediction...')
        yhat_point = ngb.predict(X_test)
        ngb_dist = ngb.pred_dist(X_test)
        yhat_dist = ngb_dist.sample(n_samples)
        # Scoring
        rmse = rmseloss_metric(yhat_point, y_test)
        crps = ps.crps_ensemble(y_test, yhat_dist.T).mean()
        # Save data
        df = df.append({'method':method, 'dataset':dataset, 'fold':fold, 'device':'cpu', 'validation_estimators': base_estimators, 'test_estimators':best_iter, 'rmse_test': rmse, 'crps_test': crps, 'validation_time':validation_time}, ignore_index=True)
#%% Save
filename = f'{method}_cpu.csv'
df.to_csv(f'pgbm/experiments/01_uci_benchmark/{filename}')