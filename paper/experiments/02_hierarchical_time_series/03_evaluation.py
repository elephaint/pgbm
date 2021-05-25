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
import pandas as pd
import numpy as np
import torch
import properscoring as ps
#%% Load data
df_lgb = pd.read_csv('pgbm/experiments/02_hierarchical_time_series/results_lightgbm_mse.csv')
df_pgbm = pd.read_csv('pgbm/experiments/02_hierarchical_time_series/results_pgbm_mse.csv')
df_ngboost = pd.read_csv('pgbm/experiments/02_hierarchical_time_series/results_ngboost_mse.csv')
df_pgbm_wmse = pd.read_csv('pgbm/experiments/02_hierarchical_time_series/results_pgbm_wmse.csv')
#%% Create levels
iteminfo = df_lgb[['date','item_id_enc', 'dept_id_enc', 'cat_id_enc']]
levels = []
levels.append(torch.from_numpy(pd.get_dummies(iteminfo['date']).values).bool())
levels.append(torch.from_numpy(pd.get_dummies(iteminfo['dept_id_enc']).values).bool())
levels.append(torch.from_numpy(pd.get_dummies(iteminfo['cat_id_enc']).values).bool())
#%% Calculate RMSE per level
def rmse_levels(yhat, y, levels):
    # Retrieve levels
    days = levels[0].T
    depts = levels[1]
    cats = levels[2]
    n_days = days.shape[0]
    n_depts = levels[1].shape[1]
    n_cats = levels[2].shape[1]
    # Create loss per level
    loss_0 = (yhat - y).pow(2).mean().sqrt()
    loss_1 = torch.zeros((n_days, n_depts))
    loss_2 = torch.zeros((n_days, n_cats))
    loss_3 = torch.zeros(n_days)
    # Loop over days    
    for day in range(n_days):
        current_day = days[day]
        yd = y[current_day]
        yhatd = yhat[current_day]
        deptd = depts[current_day]
        catd = cats[current_day]
        # Level 1
        level_1yd = (yd[:, None] * deptd).sum(0)
        level_1yhatd = (yhatd[:, None] * deptd).sum(0)
        loss_1[day] = (level_1yhatd - level_1yd).pow(2)
        # Level 2
        level_2yd = (yd[:, None] * catd).sum(0)
        level_2yhatd = (yhatd[:, None] * catd).sum(0)
        loss_2[day] = (level_2yhatd - level_2yd).pow(2)        
        # Level 3
        level_3yd = yd.sum(0)
        level_3yhatd = yhatd.sum(0)
        loss_3[day] = (level_3yhatd - level_3yd).pow(2) 
    
    loss = (loss_0, loss_1.mean().sqrt(), loss_2.mean().sqrt(), loss_3.mean().sqrt())
    
    return loss
#%% LightGBM-MSE
yhat_lgb_mse = torch.from_numpy(df_lgb['yhat_lgb'].values)
y_lgb_mse = torch.from_numpy(df_lgb['y'].values)
rmse_lgb_mse = rmse_levels(yhat_lgb_mse, y_lgb_mse, levels)    
#%% PGBM-MSE
yhat_pgbm_mse = torch.from_numpy(df_pgbm['yhat_pgbm_mu'].values)
y_pgbm_mse = torch.from_numpy(df_pgbm['y'].values)
rmse_pgbm_mse = rmse_levels(yhat_pgbm_mse, y_pgbm_mse, levels)  
#%% NGBoost-MSE
yhat_ngboost_mse = torch.from_numpy(df_ngboost['yhat_ngboost_mu'].values)
y_ngboost_mse = torch.from_numpy(df_ngboost['y'].values)
rmse_ngboost_mse = rmse_levels(yhat_ngboost_mse, y_ngboost_mse, levels)  
#%% PGBM-wMSE
yhat_pgbm_wmse = torch.from_numpy(df_pgbm_wmse['yhat_pgbm_mu'].values)
y_pgbm_wmse = torch.from_numpy(df_pgbm_wmse['y'].values)
rmse_pgbm_wmse = rmse_levels(yhat_pgbm_wmse, y_pgbm_wmse, levels)  
#%% Calculate CRPS per level
def crps_levels(yhat_dist, y, levels):
    # Retrieve levels
    days = levels[0].T
    depts = levels[1]
    cats = levels[2]
    n_days = days.shape[0]
    n_depts = levels[1].shape[1]
    n_cats = levels[2].shape[1]
    # Create loss per level
    loss_0 = ps.crps_ensemble(y, yhat_dist).mean()
    loss_1 = np.zeros((n_days, n_depts))
    loss_2 = np.zeros((n_days, n_cats))
    loss_3 = np.zeros(n_days)
    # Loop over days    
    for day in range(n_days):
        current_day = days[day]
        yd = y[current_day]
        yhatd = yhat_dist[current_day]
        deptd = depts[current_day]
        catd = cats[current_day]
        # Level 1
        level_1yd = (yd[:, None] * deptd).sum(0)
        level_1yhatd = (yhatd[:, None, :] * deptd[:, :, None]).sum(0)
        loss_1[day] = ps.crps_ensemble(level_1yd, level_1yhatd)
        # Level 2
        level_2yd = (yd[:, None] * catd).sum(0)
        level_2yhatd = (yhatd[:, None, :] * catd[:, :, None]).sum(0)
        loss_2[day] = ps.crps_ensemble(level_2yd, level_2yhatd)       
        # Level 3
        level_3yd = yd.sum(0)
        level_3yhatd = yhatd.sum(0)
        loss_3[day] = ps.crps_ensemble(level_3yd, level_3yhatd)  
    
    loss = (loss_0, loss_1.mean(), loss_2.mean(), loss_3.mean())
    
    return loss
#%% PGBM-MSE
yhat_dist_pgbm_mse = torch.from_numpy(df_pgbm.iloc[:,6:].values)
y_pgbm_mse = torch.from_numpy(df_pgbm['y'].values)
crps_pgbm_mse = crps_levels(yhat_dist_pgbm_mse, y_pgbm_mse, levels)
#%% NGBoost-MSE
yhat_dist_ngboost_mse = torch.from_numpy(df_ngboost.iloc[:,6:].values)
y_ngboost_mse = torch.from_numpy(df_ngboost['y'].values)
crps_ngboost_mse = crps_levels(yhat_dist_ngboost_mse, y_ngboost_mse, levels)
#%% PGBM-wMSE
yhat_dist_pgbm_wmse = torch.from_numpy(df_pgbm_wmse.iloc[:,6:].values)
y_pgbm_wmse = torch.from_numpy(df_pgbm_wmse['y'].values)
crps_pgbm_wmse = crps_levels(yhat_dist_pgbm_wmse, y_pgbm_wmse, levels)