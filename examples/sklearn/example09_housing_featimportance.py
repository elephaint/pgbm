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
from pgbm.sklearn import (
    HistGradientBoostingRegressor,
    crps_ensemble
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import numpy as np
import shap
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Train pgbm
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Train on set 
model = HistGradientBoostingRegressor()  
model.fit(X_train, y_train)
#% Point and probabilistic predictions
yhat_point, yhat_std = model.predict(X_test, return_std=True)
yhat_dist = model.sample(yhat_point, yhat_std, n_estimates=100, random_state=1)
# Base RMSE & CRPS on test set
rmse = np.sqrt(mean_squared_error(yhat_point, y_test))
crps = crps_ensemble(yhat_dist, y_test)         
# Print final scores
print(f'RMSE PGBM: {rmse:.2f}')
print(f'CRPS PGBM: {crps:.2f}')
#%% Feature importance with Shapley values
# NB: unfortunately does not (yet) work with shap's TreeExplainer
explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)
#%% Visualize
shap.plots.waterfall(shap_values[0])