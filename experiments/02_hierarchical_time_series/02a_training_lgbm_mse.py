# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import matplotlib.pyplot as plt # for plotting pictures
import numpy as np
import time
import lightgbm as lgb
#%% Load data
data = pd.read_hdf('pgbm/datasets/m5/m5_dataset_products.h5', key='data')
# Remove last 28 days for now...
data = data[data.date <=  '22-05-2016']
data = data[data.weeks_on_sale > 0]
data = data[data.date >= '2014-01-01']
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
params = {'min_split_gain':0,
          'min_data_in_leaf':1,
          'max_depth':-1,
          'max_bin':1024,
          'max_leaves':16,
          'learning_rate':0.1,
          'n_estimators':1000,
          'verbose':2,
          'feature_fraction':0.7,
          'bagging_fraction':0.7,
          'bagging_freq':1,
          'seed':1,
          'lambda':1,
          'objective':'rmse',
          'metric':'rmse',
          'device':'cpu'} 
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
# Train
start = time.perf_counter()  
params['bin_construct_sample_cnt'] = len(X_train)
train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_val, y_val)
model = lgb.train(params, train_set, valid_sets=[train_set, valid_set], early_stopping_rounds=20)
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
# Train
train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_test, y_test)
params['n_estimators'] = model.best_iteration + 1
start = time.perf_counter()  
params['bin_construct_sample_cnt'] = len(X_train)
model = lgb.train(params, train_set)
end = time.perf_counter()
print(f'Training time: {end - start:.2f}s')
# Save model
model.save_model('pgbm/experiments/02_hierarchical_time_series/lgbm_mse')
# Predict 
start = time.perf_counter()
yhat = model.predict(X_test)
end = time.perf_counter()
print(f'Prediction time: {end - start:.2f}s')
#%% RMSE
def rmseloss_metric(yhat, y):
    loss = np.sqrt(np.mean((yhat - y)**2))
    return loss   

y = y_test.values.squeeze()
error = rmseloss_metric(yhat, y)
#%% Save
df = pd.DataFrame({'y':y, 'yhat_lgb':yhat})
df = pd.concat((iteminfo.reset_index(drop=True), df), axis=1)
filename_day = 'pgbm/experiments/02_hierarchical_time_series/results_lightgbm_mse.csv'
df.to_csv(filename_day, index=False)
