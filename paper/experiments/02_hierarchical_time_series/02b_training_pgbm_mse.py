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
#%% Import pacakges
import pandas as pd
import numpy as np
import time
from pgbm import PGBM
import torch
#%% Load data
data = pd.read_hdf('datasets/m5/m5_dataset_products.h5', key='data')
# Remove last 28 days for now...
data = data[data.date <=  '22-05-2016']
data = data[data.weeks_on_sale > 0]
data = data[data.date >= '2013-01-01']
data = data.reset_index(drop=True)
# Choose single store
store_id = 0
subset = data[data.store_id_enc == store_id]
subset = subset.reset_index(drop=True)
#%% Preprocessing for forecast
cols_unknown = ['sales_lag1', 'sales_lag2',
   'sales_lag3', 'sales_lag4', 'sales_lag5', 'sales_lag6', 'sales_lag7',
   'sales_lag1_mavg7', 'sales_lag1_mavg28', 'sales_lag1_mavg56',
   'sales_lag7_mavg7', 'sales_lag7_mavg28', 'sales_lag7_mavg56',
   'sales_short_trend', 'sales_long_trend', 'sales_year_trend',
   'sales_item_long_trend', 'sales_item_year_trend']

cols_known = ['date','item_id_enc', 'dept_id_enc', 'cat_id_enc',
   'snap_CA',
   'snap_TX', 'snap_WI', 'event_name_1_enc', 'event_type_1_enc',
   'event_name_2_enc', 'event_type_2_enc', 'sell_price',
   'sell_price_change', 'sell_price_norm_item', 'sell_price_norm_dept',
   'weeks_on_sale', 'dayofweek_sin', 'dayofweek_cos', 'dayofmonth_sin',
   'dayofmonth_cos', 'weekofyear_sin', 'weekofyear_cos', 'monthofyear_sin',
   'monthofyear_cos', 'sales_lag364', 'sales_lag28_mavg7',
   'sales_lag28_mavg28', 'sales_lag28_mavg56', 'sales_lywow_trend',
   'sales_lag28', 'sales_lag56']

def create_forecastset(data, cols_unknown, cols_known, forecast_day):
    X_unknown = data.groupby(['store_id_enc','item_id_enc'])[cols_unknown].shift(forecast_day)
    X_known = data[cols_known]
    X = pd.concat((X_known, X_unknown), axis=1)
    y = data[['date','sales']]
    
    return X, y
#%% Set training parameters
def objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)
    
    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss

params = {'min_split_gain':0,
          'min_data_in_leaf':1,
          'max_bin':1024,
          'max_leaves':64,
          'learning_rate':0.1,
          'n_estimators':1000,
          'verbose':2,
          'feature_fraction':0.7,
          'bagging_fraction':0.7,
          'seed':1,
          'reg_lambda':1,
          'early_stopping_rounds':20,
          'tree_correlation':0.03,
          'device':'gpu',
          'gpu_device_id':0,
          'derivatives':'exact',
          'distribution':'normal'} 
#%% Validation loop
forecast_day = 0
X, y = create_forecastset(subset, cols_unknown, cols_known, forecast_day)
train_last_date = '2016-03-27'
val_first_date = '2016-03-28'
val_last_date = '2016-04-24'
X_train, y_train = X[X.date <= train_last_date], y[y.date <= train_last_date]
X_train, y_train = X_train.drop(columns='date'), y_train.drop(columns='date')
X_val, y_val = X[(X.date >= val_first_date) & (X.date <= val_last_date)], y[(y.date >= val_first_date) & (y.date <= val_last_date)]
X_val, y_val = X_val.drop(columns='date'), y_val.drop(columns='date')
train_data = (X_train, y_train)
valid_data = (X_val, y_val)
# Train
model = PGBM()
start = time.perf_counter()  
model.train(train_data, objective=objective, metric=rmseloss_metric, valid_set=valid_data, params=params)
end = time.perf_counter()
print(f'Training time: {end - start:.2f}s')
#%% Test loop
forecast_day = 0
X, y = create_forecastset(subset, cols_unknown, cols_known, forecast_day)
train_last_date = '2016-04-24'
test_first_date = '2016-04-25'
X_train, y_train = X[X.date <= train_last_date], y[y.date <= train_last_date]
X_train, y_train = X_train.drop(columns='date'), y_train.drop(columns='date')
X_test, y_test = X[X.date >= test_first_date], y[y.date >= test_first_date]
iteminfo = X_test[['date','item_id_enc', 'dept_id_enc', 'cat_id_enc']]
X_test, y_test = X_test.drop(columns='date'), y_test.drop(columns='date')
train_data = (X_train, y_train)
test_data = (X_test, y_test)
# Train
params['n_estimators'] = model.best_iteration
model = PGBM()
start = time.perf_counter()  
model.train(train_data, objective=objective, metric=rmseloss_metric, params=params)
end = time.perf_counter()
print(f'Training time: {end - start:.2f}s')
# Save model
model.save('experiments/02_hierarchical_time_series/pgbm_mse.model')
# Predict 
start = time.perf_counter()
yhat = model.predict(X_test)
model.params['tree_correlation'] = np.log10(len(X_train)) / 100
yhat_dist = model.predict_dist(test_data[0], n_forecasts=1000)
end = time.perf_counter()
print(f'Prediction time: {end - start:.2f}s')
#%% RMSE
error = rmseloss_metric(yhat.cpu(), test_data[1].values.squeeze())
print(error)
#%% Save
df = pd.DataFrame({'y':test_data[1].values.squeeze(), 'yhat_pgbm_mu':yhat.cpu()})
df = pd.concat((df, pd.DataFrame(yhat_dist.cpu().clamp(0).numpy().T)), axis=1)
df = pd.concat((iteminfo.reset_index(drop=True), df), axis=1)
filename_day = 'experiments/02_hierarchical_time_series/results_pgbm_mse.csv'
df.to_csv(filename_day, index=False)
