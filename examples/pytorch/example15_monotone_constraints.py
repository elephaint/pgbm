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
from pgbm import PGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
#%% Load data
# The Boston dataset has 14 features (see here: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
# We want the first feature (CRIM: per capita crime rate by town) to have a negative monotonicity with the output.
# We want the sixth feature (RM: average number of rooms per dwelling) to have a positive monotonicity with the output.
# Note that these are just our assumptions! This needn't be necessarily true, we would just like to enforce these assumptions in our model.
X, y = load_boston(return_X_y=True)
#%% First, we train pgbm without monotonicity constraints
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
train_data = (X_train, y_train)
#%% Compute the dependency. We can compute this on train and test set, for speed purposes we choose test_set here.
model = PGBMRegressor().fit(X_train, y_train)
y0, x0 = partial_dependence(model, X_test, [0])
y5, x5 = partial_dependence(model, X_test, [5])
#%% Plot dependencies. 
# Note how the relationships are non-linear for both features.
fig, ax = plt.subplots(1, 2)
ax[0].plot(x0[0], y0.squeeze())
ax[0].set_title('Dependency plot of feature 0')
ax[0].set(xlabel = 'Feature value', ylabel='Output')

ax[1].plot(x5[0], y5.squeeze())
ax[1].set_title('Dependency plot of feature 5')
ax[1].set(xlabel = 'Feature value', ylabel='Output')
#%% Set monotonicity parameters: +1 for positive slope, -1 for negative slope, 0 for no constraint. 
# Increase monotone_iterations to improve accuracy at the cost of speed.
monotone_constraints = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
model = PGBMRegressor(monotone_iterations=5, monotone_constraints = monotone_constraints).fit(X_train, y_train)
y0, x0 = partial_dependence(model, X_test, [0])
y5, x5 = partial_dependence(model, X_test, [5])
#%% Plot dependencies. 
# Note how the relationships are now linear for both features - decreasing for the first and increasing for the sixth feature.
fig, ax = plt.subplots(1, 2)
ax[0].plot(x0[0], y0.squeeze())
ax[0].set_title('Dependency plot of feature 0')
ax[0].set(xlabel = 'Feature value', ylabel='Output')

ax[1].plot(x5[0], y5.squeeze())
ax[1].set_title('Dependency plot of feature 5')
ax[1].set(xlabel = 'Feature value', ylabel='Output')
