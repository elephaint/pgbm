
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
from sklearn.utils.estimator_checks import parametrize_with_checks
from pgbm.sklearn import HistGradientBoostingRegressor
from pgbm.torch import PGBMRegressor

@parametrize_with_checks([HistGradientBoostingRegressor(),
                        PGBMRegressor(verbose=0)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
