
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
#%%
import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
from pgbm.sklearn import crps_ensemble
from pgbm.torch import PGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load data
X, y = fetch_california_housing(return_X_y=True)
#%%
# Estimator checks scikit-learn
@parametrize_with_checks([PGBMRegressor(verbose=0)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

# Train on housing with default parameters - torch
@pytest.mark.parametrize("X, y", [(X, y)])
def test_train_housing_torch(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    model = PGBMRegressor(random_state=0, verbose=0)
    model.fit(X_train, y_train)
    yhat_point = model.predict(X_test)
    yhat_dist = model.predict_dist(X_test, n_forecasts=1000)
    rmse = np.sqrt(mean_squared_error(yhat_point, y_test))
    crps = crps_ensemble(yhat_dist, y_test)
    rmse_round = np.round(rmse, 2)
    crps_round = np.round(crps, 2)
    assert rmse_round == 0.47
    assert crps_round == 0.24

@pytest.mark.parametrize("X, y", [(X, y)])
def test_nans(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    # Add nans
    nans = np.zeros([X_train.shape[0],1])
    nans[:,:] = np.nan
    X_train = np.append(X_train, nans, axis=1)
    model = PGBMRegressor().fit(X_train, y_train)  