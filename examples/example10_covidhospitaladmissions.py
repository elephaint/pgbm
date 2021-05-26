# -*- coding: utf-8 -*-
"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam & Kickstart.AI

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
from pgbm import PGBM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%% Load data
# Cases
df = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_ziekenhuisopnames.csv', sep=';', parse_dates=[0], usecols=[2, 3, 4, 5, 6, 8])
# Proportional allocation of NaNs per municipality per date
total_by_date = df.groupby(['Date_of_statistics'])['Hospital_admission'].sum()
total_by_date_without_nan = df.groupby(['Date_of_statistics','Municipality_code'])['Hospital_admission'].sum().groupby(['Date_of_statistics']).sum()
df_totals = pd.concat((total_by_date, total_by_date_without_nan), axis=1)
df_totals.columns = ['Hospital_admission_total_correct', 'Hospital_admission_total_partial']
df_totals['Delta_hospital_admissions'] = df_totals['Hospital_admission_total_correct'] - df_totals['Hospital_admission_total_partial']
df_totals = df_totals.reset_index()
df = df.merge(df_totals, how='left')
df['Hospital_admission'] = df['Hospital_admission'] + (df['Hospital_admission'] / df['Hospital_admission_total_partial']) * df['Delta_hospital_admissions']
df = df.drop(columns = ['Hospital_admission_total_correct', 'Hospital_admission_total_partial','Delta_hospital_admissions'])
df = df.groupby(['Municipality_name', 'Security_region_name','Municipality_code','Security_region_code','Date_of_statistics']).sum()
df = df.reset_index()
# Remove last date, data for those days seems incomplete
last_date = df['Date_of_statistics'].unique()[-1]
df = df[df['Date_of_statistics'] < last_date]
df = df.reset_index(drop=True)
#%% Merge test data
df_tests = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_uitgevoerde_testen.csv', sep=';', parse_dates=[0], usecols=[2,3,5,6])
df_tests['positivity_rate'] =  df_tests['Tested_positive'] / df_tests['Tested_with_result']
df = df.merge(df_tests, how='left', left_on=['Security_region_code', 'Date_of_statistics'], right_on=['Security_region_code', 'Date_of_statistics'])
#%% Merge ROAZ region
df_cases = pd.read_csv('https://data.rivm.nl/covid-19/COVID-19_aantallen_gemeente_per_dag.csv', sep=';', usecols=[2, 8]).drop_duplicates().dropna()
index_to_drop = df_cases[(df_cases.Municipality_code == 'GM0363') & (df_cases.ROAZ_region == 'Netwerk Acute Zorg Noordwest')].index
df_cases = df_cases.drop(index_to_drop)
df = df.merge(df_cases, how='left', left_on=['Municipality_code'], right_on=['Municipality_code'])
#%% Label encoding
from sklearn.preprocessing import LabelEncoder
categorical_columns = ['Municipality_code', 'Security_region_code', 'ROAZ_region']
for col in categorical_columns:
    df[col+'_enc'] = LabelEncoder().fit_transform(df[col].fillna('Null'))
    df[col+'_enc'] = df[col+'_enc'].astype('int16')

df = df.drop(columns = ['Municipality_code', 'Security_region_code'])
#%% Add time indicators
df['day_of_week'] = df['Date_of_statistics'].dt.dayofweek
df['day_of_month'] = df['Date_of_statistics'].dt.day
#%% Sort by municipality code, then by date
df = df.sort_values(by=['Municipality_code_enc','Date_of_statistics'])
df = df.reset_index(drop=True)
#%% Add lags + moving averages
# Add rolling target
window = 7
group = df.groupby(['Municipality_code_enc'])['Hospital_admission']
df['Hospital_admission_rolling'] = group.transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0).astype('float32')
# Rolling positive tests
group = df.groupby(['Municipality_code_enc'])['Tested_positive']
df['Tested_positive_rolling'] = group.transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0).astype('float32')

# Lag target
lags = [window, 2 * window, 3 * window, 4 * window]
for lag in lags:
    group = df.groupby(['Municipality_code_enc'])['Hospital_admission']
    df['Hospital_admission_lag'+str(lag)+'d'] = group.shift(lag)
    group = df.groupby(['Municipality_code_enc'])['Tested_positive']
    df['Tested_positive_lag'+str(lag)+'d'] = group.shift(lag)

# Short-term target growth    
df['Hospital_admission_growth'] = df['Hospital_admission'] / df['Hospital_admission_rolling'] - 1  
df['Tested_positive_growth'] = df['Tested_positive'] / df['Tested_positive_lag7d'] - 1
#%% Set target variable
forecast_period = 7
df['y'] = df.groupby(['Municipality_code_enc'])['Hospital_admission'].shift(-forecast_period)
#%% Drop too old dates
df = df[df['Date_of_statistics'] >= '2020-09-01']
df = df.reset_index(drop=True)
#%% CRPS per level
def crps_levels(yhat_dist, y, levels):
    # Retrieve levels
    days = levels[0].T
    n_days = days.shape[0]
    loss = torch.zeros((len(levels) + 1, 1), device=yhat_dist.device)
    # First loss is CRPS over each sample
    loss[0] = model.crps_ensemble(yhat_dist, y).mean()
    # Loop over days. For each level there is a daily loss. The final level is the daily aggregate loss    
    for day in range(n_days):
        current_day = days[day]
        yd = y[current_day]
        yhatd = yhat_dist[:, current_day]
        for i, level in enumerate(levels[1:]):
            level_day = level[current_day].float()
            level_yd = torch.einsum('i, ij -> j', yd, level_day)
            level_yhatd = torch.einsum('ki, ij -> kj', yhatd, level_day)
            crps_level_day = model.crps_ensemble(level_yhatd, level_yd)
            loss[i + 1] += crps_level_day.mean() / n_days

        level_fyd = yd.sum(0, keepdim=True)
        level_fyhatd = yhatd.sum(1, keepdim=True)
        loss[-1] += model.crps_ensemble(level_fyhatd, level_fyd) / n_days  
       
    return loss.squeeze()

#%% Validation: 8 weeks rolling
n_validation_weeks = 8
last_date = df['Date_of_statistics'].max()
validation_dates = [last_date - pd.Timedelta((i + 1) * 7, 'D') for i in range(n_validation_weeks, 0, -1)]

# Parameters
device = torch.device(0)

params = {'min_split_gain':0,
      'min_data_in_leaf':1,
      'max_leaves':8,
      'max_bin':64,
      'learning_rate':0.1,
      'n_estimators':2000,
      'verbose':2,
      'early_stopping_rounds':100,
      'feature_fraction':1,
      'bagging_fraction':1,
      'seed':1,
      'lambda':1,
      'tree_correlation':0.0,
      'device':'gpu',
      'output_device':'gpu',
      'gpu_device_ids':(0,),
      'derivatives':'exact',
      'distribution':'negativebinomial'} 

# Number of samples for distribution
n_samples = 100

# Objective
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = yhat*0. + 1

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss

# Validation loop: validate for n_validation_weeks, and across a choice of distributions & tree correlations
df_val_result = pd.DataFrame()
tree_correlations = np.arange(30, step=2) * 0.01
distributions = ['poisson', 'weibull', 'gumbel', 'gamma', 'negativebinomial']
for fold, date in enumerate(validation_dates):
    df_train = df[df.Date_of_statistics <= date].copy()
    df_val = df[(df.Date_of_statistics > date) & (df.Date_of_statistics <= date + pd.Timedelta(7, 'D'))].copy()    
    iteminfo_val = df_val[['Date_of_statistics','Security_region_code_enc','ROAZ_region_enc']]
    df_train.drop(columns=['Date_of_statistics','Municipality_name', 'Security_region_name', 'ROAZ_region'], inplace=True)
    df_val.drop(columns=['Date_of_statistics','Municipality_name', 'Security_region_name', 'ROAZ_region'], inplace=True)
    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1] 
    X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1] 
    # Create levels for hierarchical forecast
    levels_val = []
    levels_val.append(torch.from_numpy(pd.get_dummies(iteminfo_val['Date_of_statistics']).values).bool().to(device))
    levels_val.append(torch.from_numpy(pd.get_dummies(iteminfo_val['Security_region_code_enc']).values).bool().to(device))    
    levels_val.append(torch.from_numpy(pd.get_dummies(iteminfo_val['ROAZ_region_enc']).values).bool().to(device))    
    # Create datasets
    torchdata = lambda x : torch.from_numpy(x.values).float()
    train_data = (torchdata(X_train), torchdata(y_train))
    val_data = (torchdata(X_val), torchdata(y_val))
    model = PGBM()
    model.train(train_data, objective=mseloss_objective, valid_set = val_data, metric=rmseloss_metric, params=params)
    # Calculate best fitting tree correlation and distribution for this fold
    crps_pgbm = np.zeros((len(tree_correlations), len(distributions)))
    for i, tree_correlation in enumerate(tree_correlations):
        for j, distribution in enumerate(distributions):
            print(f'Fold {fold} / Tree correlation {i} / Distribution {j}')
            model.params['tree_correlation'] = tree_correlation
            model.params['distribution'] = distribution
            yhat_dist = model.predict_dist(val_data[0], n_samples=n_samples)
            yhat_dist = yhat_dist.clamp(0, 1e9)  # clipping to avoid negative predictions of symmetric distributions
            crps = crps_levels(yhat_dist, val_data[1].to(device), levels_val)
            df_val_result = df_val_result.append({'fold': fold, 'tree_correlation': tree_correlation, 'distribution': distribution, 'best_iteration': model.best_iteration, 'best_score': model.best_score.cpu().numpy().item(), 'crps_municipality': crps[0].cpu().numpy().item(), 'crps_security_region': crps[1].cpu().numpy().item(), 'crps_roaz_region': crps[2].cpu().numpy().item(), 'crps_national': crps[3].cpu().numpy().item()}, ignore_index=True)

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%I%M%S")
# Set number of iterations to median of validation estimators
params['n_estimators'] = int(df_val_result['best_iteration'].median())
# Set best fitting distribution and tree correlation according to validation result
df_val_crps = df_val_result.groupby(['tree_correlation', 'distribution'])[['crps_municipality','crps_security_region','crps_roaz_region', 'crps_national']].mean()
params['tree_correlation'] = df_val_crps.sum(1).idxmin()[0]
params['distribution'] = df_val_crps.sum(1).idxmin()[1]
#%% Train on full dataset
# Select training data
test_date = validation_dates[-1] + pd.Timedelta(7, 'D')
df_train = df[df.Date_of_statistics <= test_date].copy()
df_test = df[df.Date_of_statistics > test_date].copy()
df_train.drop(columns=['Date_of_statistics','Municipality_name', 'Security_region_name', 'ROAZ_region'], inplace=True)
df_test.drop(columns=['Date_of_statistics','Municipality_name', 'Security_region_name', 'ROAZ_region'], inplace=True)
X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1] 
X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1] 
train_data = (torchdata(X_train), torchdata(y_train))
# Train
model = PGBM()
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric, params=params)
# Predict
yhat_test = model.predict(X_test).cpu().numpy()
yhat_test_dist = model.predict_dist(X_test, n_samples=n_samples)
yhat_test_dist = np.clip(yhat_test_dist.cpu().numpy(), 0, 1e9)
# Add results to df
df_pred_test = pd.DataFrame(index=y_test.index, data=yhat_test, columns=['Hospital_admission_predicted'])
col_names = ['sample_'+str(i) for i in range(n_samples)]
df_pred_test_dist = pd.DataFrame(index=y_test.index, data=yhat_test_dist.T, columns=col_names)
#%% Store predictions
df_result = df[['Date_of_statistics', 'Municipality_name', 'Security_region_name', 'ROAZ_region']].copy()
df_result['Date_forecast_created'] = pd.Timestamp.now()
df_result['Date_prediction'] = df_result['Date_of_statistics'] + pd.Timedelta(7, 'D')
df_result = df_result[['Date_prediction', 'Date_forecast_created', 'Date_of_statistics', 'Municipality_name', 'Security_region_name', 'ROAZ_region']]
df_result = pd.concat((df_result, df_pred_test, df_pred_test_dist), axis=1)
df_result = df_result[df_result.Date_of_statistics > test_date].copy().reset_index(drop=True)
#%% Plot all
group = df_result.groupby(['Date_of_statistics'])
aggregate_samples = group.sum()
quantiles = [0.1, 0.9]
quantile_values = np.quantile(aggregate_samples.values, quantiles, axis=1)

fig, ax1 = plt.subplots()
ax1.plot(df.groupby(['Date_of_statistics'])['Hospital_admission'].sum(), label='Actual')
ax1.plot(group['Hospital_admission_predicted'].sum().index + pd.Timedelta(7, 'D'), group['Hospital_admission_predicted'].sum(), 'k--', label='Prediction')
ax1.fill_between(group['Hospital_admission_predicted'].sum().index + pd.Timedelta(7, 'D'), quantile_values[0], quantile_values[1], color="#b9cfe7", label=f'P{quantiles[0]*100}-P{quantiles[1]*100} confidence interval')
ax1.title.set_text('Hospital admissions - national level forecast')
fig.legend(loc='upper right')
