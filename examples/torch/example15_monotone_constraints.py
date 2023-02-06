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

#%% Load packages, we use our sklearn wrapper to make use of sklearn's helper functions
from pgbm.torch import PGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
#%% Load data
# For the California housing dataset, we expect a positive monotonicity constraint 
# on (at least) the first and third feature, representing respectively the median income
# and average number of rooms per household
X, y = fetch_california_housing(return_X_y=True)
#%% First, we train pgbm without monotonicity constraints
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#%% Compute the dependency. We can compute this on train and test set, for speed purposes we choose test_set here.
model = PGBMRegressor().fit(X_train, y_train)
PartialDependenceDisplay.from_estimator(model, X_test, [0])
PartialDependenceDisplay.from_estimator(model, X_test, [2])
#%% Set monotonicity parameters: +1 for positive slope, -1 for negative slope, 0 for no constraint. 
# Increase monotone_iterations to improve accuracy at the cost of speed.
monotone_constraints = [1, 0, 1, 0, 0, 0, 0, 0]
model = PGBMRegressor(monotone_iterations=5, monotone_constraints = monotone_constraints).fit(X_train, y_train)
PartialDependenceDisplay.from_estimator(model, X_test, [0])
PartialDependenceDisplay.from_estimator(model, X_test, [2])