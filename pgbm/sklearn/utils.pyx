"""This module contains utility routines."""
# Author: Nicolas Hug

from cython.parallel import prange
cimport numpy as cnp
cnp.import_array()

from sklearn.base import is_classifier
from .binning import _BinMapper
ctypedef cnp.npy_float32 G_H_DTYPE_C
ctypedef cnp.npy_float64 Y_DTYPE_C

def get_equivalent_estimator(estimator, lib='lightgbm', n_classes=None):
    """Return an unfitted estimator from another lib with matching hyperparams.

    This utility function takes care of renaming the sklearn parameters into
    their LightGBM, XGBoost or CatBoost equivalent parameters.

    # unmapped XGB parameters:
    # - min_samples_leaf
    # - min_data_in_bin
    # - min_split_gain (there is min_split_loss though?)

    # unmapped Catboost parameters:
    # max_leaves
    # min_*
    """

    if lib not in ('lightgbm', 'xgboost', 'catboost'):
        raise ValueError('accepted libs are lightgbm, xgboost, and catboost. '
                         ' got {}'.format(lib))

    sklearn_params = estimator.get_params()

    if sklearn_params['loss'] == 'auto':
        raise ValueError('auto loss is not accepted. We need to know if '
                         'the problem is binary or multiclass classification.')
    if sklearn_params['early_stopping']:
        raise NotImplementedError('Early stopping should be deactivated.')

    lightgbm_loss_mapping = {
        'squared_error': 'regression_l2',
        'absolute_error': 'regression_l1',
        'log_loss': 'binary' if n_classes == 2 else 'multiclass',
    }

    lightgbm_params = {
        'objective': lightgbm_loss_mapping[sklearn_params['loss']],
        'learning_rate': sklearn_params['learning_rate'],
        'n_estimators': sklearn_params['max_iter'],
        'num_leaves': sklearn_params['max_leaf_nodes'],
        'max_depth': sklearn_params['max_depth'],
        'min_child_samples': sklearn_params['min_samples_leaf'],
        'reg_lambda': sklearn_params['l2_regularization'],
        'max_bin': sklearn_params['max_bins'],
        'min_data_in_bin': 1,
        'min_child_weight': 1e-3,
        'min_sum_hessian_in_leaf': 1e-3,
        'min_split_gain': 0,
        'verbosity': 10 if sklearn_params['verbose'] else -10,
        'boost_from_average': True,
        'enable_bundle': False,  # also makes feature order consistent
        'subsample_for_bin': _BinMapper().subsample,
    }

    if sklearn_params['loss'] == 'log_loss' and n_classes > 2:
        # LightGBM multiplies hessians by 2 in multiclass loss.
        lightgbm_params['min_sum_hessian_in_leaf'] *= 2
        # LightGBM 3.0 introduced a different scaling of the hessian for the multiclass case.
        # It is equivalent of scaling the learning rate.
        # See https://github.com/microsoft/LightGBM/pull/3256.
        if n_classes is not None:
            lightgbm_params['learning_rate'] *= n_classes / (n_classes - 1)

    # XGB
    xgboost_loss_mapping = {
        'squared_error': 'reg:linear',
        'absolute_error': 'LEAST_ABSOLUTE_DEV_NOT_SUPPORTED',
        'log_loss': 'reg:logistic' if n_classes == 2 else 'multi:softmax',
    }

    xgboost_params = {
        'tree_method': 'hist',
        'grow_policy': 'lossguide',  # so that we can set max_leaves
        'objective': xgboost_loss_mapping[sklearn_params['loss']],
        'learning_rate': sklearn_params['learning_rate'],
        'n_estimators': sklearn_params['max_iter'],
        'max_leaves': sklearn_params['max_leaf_nodes'],
        'max_depth': sklearn_params['max_depth'] or 0,
        'lambda': sklearn_params['l2_regularization'],
        'max_bin': sklearn_params['max_bins'],
        'min_child_weight': 1e-3,
        'verbosity': 2 if sklearn_params['verbose'] else 0,
        'silent': sklearn_params['verbose'] == 0,
        'n_jobs': -1,
    }

    # Catboost
    catboost_loss_mapping = {
        'squared_error': 'RMSE',
        # catboost does not support MAE when leaf_estimation_method is Newton
        'absolute_error': 'LEAST_ASBOLUTE_DEV_NOT_SUPPORTED',
        'log_loss': 'Logloss' if n_classes == 2 else 'MultiClass',
    }

    catboost_params = {
        'loss_function': catboost_loss_mapping[sklearn_params['loss']],
        'learning_rate': sklearn_params['learning_rate'],
        'iterations': sklearn_params['max_iter'],
        'depth': sklearn_params['max_depth'],
        'reg_lambda': sklearn_params['l2_regularization'],
        'max_bin': sklearn_params['max_bins'],
        'feature_border_type': 'Median',
        'leaf_estimation_method': 'Newton',
        'verbose': bool(sklearn_params['verbose']),
    }

    if lib == 'lightgbm':
        from lightgbm import LGBMRegressor
        from lightgbm import LGBMClassifier
        if is_classifier(estimator):
            return LGBMClassifier(**lightgbm_params)
        else:
            return LGBMRegressor(**lightgbm_params)

    elif lib == 'xgboost':
        from xgboost import XGBRegressor
        from xgboost import XGBClassifier
        if is_classifier(estimator):
            return XGBClassifier(**xgboost_params)
        else:
            return XGBRegressor(**xgboost_params)

    else:
        from catboost import CatBoostRegressor
        from catboost import CatBoostClassifier
        if is_classifier(estimator):
            return CatBoostClassifier(**catboost_params)
        else:
            return CatBoostRegressor(**catboost_params)


def sum_parallel(G_H_DTYPE_C [:] array, int n_threads):

    cdef:
        Y_DTYPE_C out = 0.
        int i = 0

    for i in prange(array.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        out += array[i]

    return out

def sum_parallel_with_squares(G_H_DTYPE_C [:] array, int n_threads):

    cdef:
        Y_DTYPE_C sum_out = 0.
        Y_DTYPE_C sum_squared_out = 0.
        int i = 0

    for i in prange(array.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        sum_out += array[i]
        sum_squared_out += array[i]**2

    return (sum_out, sum_squared_out)

def sum_parallel_with_squares_two_arrays(G_H_DTYPE_C [:] array_x,
        G_H_DTYPE_C [:] array_y, int n_threads):

    cdef:
        Y_DTYPE_C sum_x_out = 0.
        Y_DTYPE_C sum_x_squared_out = 0.
        Y_DTYPE_C sum_y_out = 0.
        Y_DTYPE_C sum_y_squared_out = 0.
        Y_DTYPE_C sum_xy_out = 0.
        int i = 0

    for i in prange(array_x.shape[0], schedule='static', nogil=True,
                    num_threads=n_threads):
        sum_x_out += array_x[i]
        sum_x_squared_out += array_x[i]**2
        sum_y_out += array_y[i]
        sum_y_squared_out += array_y[i]**2
        sum_xy_out += array_x[i] * array_y[i]

    return (sum_x_out, sum_y_out, sum_x_squared_out, sum_y_squared_out, sum_xy_out)