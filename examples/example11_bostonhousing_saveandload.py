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
import torch
from pgbm import PGBM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#%% Objective for pgbm
def mseloss_objective(yhat, y, levels=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y, levels=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = load_boston(return_X_y=True)
#%% Train pgbm and save
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_data = (X_train, y_train)
# Train on set 
model = PGBM()  
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
model.save('model.pt')
#%% Load model trained with PyTorch-CPU and predict with Pytorch-CPU
model_new = PGBM()
model_new.load('model.pt')
#% Point and probabilistic predictions
yhat_point = model_new.predict(X_test)
yhat_dist = model_new.predict_dist(X_test)
# Scoring
crps = model_new.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'CRPS PGBM: {crps:.2f}')
#%% Load model trained with PyTorch-CPU and predict with Pytorch-GPU
model_new = PGBM()
model_new.load('model.pt', torch.device(0))
#% Point and probabilistic predictions
yhat_point = model_new.predict(X_test)
yhat_dist = model_new.predict_dist(X_test)
# Scoring
crps = model_new.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'CRPS PGBM: {crps:.2f}')
#%% Load model trained with PyTorch-CPU and predict with Numba backend.
from pgbm_nb import PGBM
model_new = PGBM()
model_new.load('model.pt')
#% Point and probabilistic predictions
yhat_point = model_new.predict(X_test)
yhat_dist = model_new.predict_dist(X_test)
# Scoring
crps = model_new.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'CRPS PGBM: {crps:.2f}')