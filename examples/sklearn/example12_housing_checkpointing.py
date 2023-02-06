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
from pgbm.sklearn import HistGradientBoostingRegressor, crps_ensemble
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import joblib
#%% Load data
X, y = fetch_california_housing(return_X_y=True)
#%% Train pgbm and save
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Train on set 
model = HistGradientBoostingRegressor(max_iter=50, verbose=2, warm_start=True)
model.fit(X_train, y_train)
joblib.dump(model, 'model.pt')
#%% Load checkpoint and continue training
model_new = joblib.load('model.pt')
model_new.set_params({'max_iter': 100})
model_new.fit(X_train, y_train)
# Scoring
yhat_point, yhat_std = model_new.predict(X_test, return_std=True)
yhat_dist = model_new.sample(yhat_point, yhat_std, n_estimates=100, random_state=1)
# Scoring
crps = crps_ensemble(yhat_dist, y_test)         
# Print final scores
print(f'CRPS PGBM: {crps:.2f}')