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

#%% Load packages
print('Testing Sklearn version')
from pgbm.sklearn import HistGradientBoostingRegressor, crps_ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
# Train on set 
model = HistGradientBoostingRegressor(random_state=0)  
model.fit(X_train, y_train)
#% Point and probabilistic predictions. By default, 1 probabilistic estimates is created, so we create 100
yhat_point, yhat_point_std = model.predict(X_test, return_std=True)
yhat_dist = model.sample(yhat_point, yhat_point_std, n_estimates=1000, random_state=0)
# Scoring
rmse = np.sqrt(mean_squared_error(yhat_point, y_test))
crps = crps_ensemble(yhat_dist, y_test)    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
#%% Load packages
print('Testing Torch version')
import torch
from pgbm.torch import PGBM
#%% Objective for pgbm
def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
train_data = (X_train, y_train)
# Train on set 
model = PGBM()  
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions. By default, 100 probabilistic estimates are created
yhat_point = model.predict(X_test)
yhat_dist = model.predict_dist(X_test)
# Scoring
rmse = model.metric(yhat_point.cpu(), y_test)
crps = model.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')