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
from pgbm import PGBM
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
#%% Objective for pgbm
def mseloss_objective(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean()

    return loss

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Set parameters to use autodifferentiation on the loss function
params = {'derivatives': 'approx'}
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_data = (X_train, y_train)
# Train on set   
model = PGBM()
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric, params=params)
#% Point and probabilistic predictions
yhat_point_pgbm = model.predict(X_test)
yhat_dist_pgbm = model.predict_dist(X_test)
# Scoring. Note: move the outputs back to CPU
rmse = model.metric(yhat_point_pgbm.cpu(), y_test)
crps = model.crps_ensemble(yhat_dist_pgbm.cpu(), y_test).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
#%% Plot all samples
plt.plot(y_test, 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.legend()