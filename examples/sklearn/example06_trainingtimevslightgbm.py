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
#%% Import packages
from pgbm.sklearn import HistGradientBoostingRegressor
import time
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from datasets import get_dataset, get_fold
#%% Load data
#datasets = ['housing', 'concrete', 'energy', 'kin8nm', 'msd', 'naval', 'power', 'protein', 'wine', 'yacht','higgs']
dataset = 'msd'
data = get_dataset(dataset)
X_train, _, y_train, _ = get_fold(dataset, data, random_state=0)
#%% Parameters
params = {'min_samples_leaf':2,
      'max_leaf_nodes':8,
      'max_bins':64,
      'learning_rate':0.1,
      'max_iter':2000,
      'verbose':2,
      'early_stopping':False,
      'random_state':1,
      'l2_regularization':1}
#%% PGBM-sklearn
start = time.perf_counter()
model = HistGradientBoostingRegressor(**params)
model.fit(X_train, y_train)
end = time.perf_counter()
print(f'Fold time: {end - start:.2f}s')
#%% LightGBM
# Additional parameters
params['objective'] = 'regression'
params['min_data_in_bin'] = 1
params['bin_construct_sample_cnt'] = len(X_train)
params['device'] = 'cpu'
params['early_stopping'] = None
# Train LightGBM
dtrain = lgb.Dataset(X_train, y_train)
start = time.perf_counter()
lgb_model = lgb.train(params, dtrain)
end = time.perf_counter()
print(f'Fold time: {end - start:.2f}s')