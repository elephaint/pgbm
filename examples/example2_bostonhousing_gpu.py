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

#%% Load packages
import torch
import pgbm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
#%% Objective for pgbm
def mseloss_objective(yhat, y):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = load_boston(return_X_y=True)
#%% Set parameters to train on GPU
params = {'device': 'gpu',
          'output_device': 'gpu',
          'gpu_device_ids': (0, )}
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Build tuples of torch datasets
torchdata = lambda x : torch.from_numpy(x).float()
train_data = (torchdata(X_train), torchdata(y_train))
test_data = (torchdata(X_test), torchdata(y_test))
# Train on set   
model = pgbm.PGBM()
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions
yhat_point_pgbm = model.predict(test_data[0])
yhat_dist_pgbm = model.predict_dist(test_data[0], n_samples=1000)
# Scoring
rmse = rmseloss_metric(yhat_point_pgbm, test_data[1])
crps = pgbm.crps_ensemble(test_data[1], yhat_dist_pgbm).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
#%% Plot all samples
plt.plot(test_data[1], 'o', label='Actual')
plt.plot(yhat_point_pgbm.cpu(), 'ko', label='Point prediction PGBM')
plt.plot(yhat_dist_pgbm.cpu().max(dim=0).values, 'k--', label='Max bound PGBM')
plt.plot(yhat_dist_pgbm.cpu().min(dim=0).values, 'k--', label='Min bound PGBM')
plt.legend()