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
import matplotlib.pyplot as plt
#%% Objective for pgbm
def mseloss_objective(yhat, y, sample_weight=None):
    gradient = (yhat - y)
    hessian = torch.ones_like(yhat)

    return gradient, hessian

def rmseloss_metric(yhat, y, sample_weight=None):
    loss = (yhat - y).pow(2).mean().sqrt()

    return loss
#%% Load data
X, y = load_boston(return_X_y=True)
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_data = (X_train, y_train)
# Train on set 
model = PGBM()  
model.train(train_data, objective=mseloss_objective, metric=rmseloss_metric)
#% Point and probabilistic predictions
yhat_point = model.predict(X_test)
yhat_dist = model.predict_dist(X_test)
# Scoring
rmse = model.metric(yhat_point, y_test)
crps = model.crps_ensemble(yhat_dist, y_test).mean()    
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
#%% Feature importance from split gain on training set
feature_names = load_boston()['feature_names']
val_fi, idx_fi = torch.sort(model.feature_importance.cpu())
#%% Feature importance from permutation importance on test set (supervised). This can be slow to calculate!
permutation_importance_supervised = model.permutation_importance(X_test, y_test)
mean_permutation_importance_supervised = permutation_importance_supervised.mean(1)
_, idx_pi_sup = torch.sort(mean_permutation_importance_supervised)
#%% Feature importance from permutation importance on test set (unsupervised). This can be slow to calculate!
permutation_importance_unsupervised = model.permutation_importance(X_test)
mean_permutation_importance_unsupervised = permutation_importance_unsupervised.mean(1)
_, idx_pi_unsup = torch.sort(mean_permutation_importance_unsupervised)
#%% Plot both
fig, ax = plt.subplots(1, 3)
ax[0].barh(feature_names[idx_fi], val_fi)
ax[0].set_title('Feature importance by cumulative split gain on training set')
ax[0].set(xlabel = 'Cumulative split gain', ylabel='Feature')

ax[1].set_title('Feature importance by feature permutation on test set (supervised)')
ax[1].boxplot(permutation_importance_supervised[idx_pi_sup], labels=feature_names[idx_pi_sup], vert=False)
ax[1].set(xlabel = '% change in error metric', ylabel='Feature')

ax[2].set_title('Feature importance by feature permutation on test set (unsupervised)')
ax[2].boxplot(permutation_importance_unsupervised[idx_pi_unsup], labels=feature_names[idx_pi_unsup], vert=False)
ax[2].set(xlabel = '% change in predictions', ylabel='Feature')

fig.tight_layout()
plt.show()