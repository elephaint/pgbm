import numpy as np
from numba import njit, prange, float64
from sklearn.metrics._scorer import _BaseScorer
from pgbm.sklearn.common import Y_DTYPE

def crps_ensemble(yhat_dist, y):
    """Calculate the empirical Continuously Ranked Probability Score (CRPS) 
    for a set of forecasts for a number of samples (lower is better). 
    
    Based on `crps_ensemble` from `properscoring` https://pypi.org/project/properscoring/
    
    :param yhat_dist: forecasts for each sample of size [n_forecasts x n_samples].
    :type yhat_dist: np.ndarray
    :param y: ground truth value of each sample of size [n_samples].
    :type y: np.ndarray
    
    :return: mean CRPS score of all samples
    :rtype: np.float64
    
    Example:
    
    .. code-block:: python
        
        train_set = (X_train, y_train)
        test_set = (X_test, y_test)
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        yhat_point, yhat_std = model.predict(X_test, return_std=True)
        yhat_test_dist = model.sample(yhat_point, yhat_std, n_estimates=100)
        crps = crps_ensemble(yhat_test_dist, y_test)
        
    """
    assert y.shape[0] == yhat_dist.shape[1], "yhat_dist should have the same number of samples as y"
    assert isinstance(yhat_dist, np.ndarray), "yhat_dist should be a Numpy array"
    assert isinstance(y, np.ndarray), "y should be a Numpy array"
    if yhat_dist.dtype != Y_DTYPE:
        yhat_dist = yhat_dist.astype(Y_DTYPE, order="F")
    if y.dtype != Y_DTYPE:
        y = y.astype(Y_DTYPE, order="C")
    yhat_dist = np.asfortranarray(yhat_dist)
    y = np.ascontiguousarray(y)
    return _crps_ensemble(yhat_dist, y)

def make_probabilistic_scorer(
    score_func,
    greater_is_better=True,
    **kwargs,
):
    """Make a probabilistic scorer from a performance metric or loss function.
    This factory function wraps the probabilistic scoring function
    crps_ensemble.

    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        `score_func(y, y_pred, **kwargs)`.
    greater_is_better : bool, default=True
        Whether `score_func` is a score function (default), meaning high is
        good, or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the `score_func`.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.

    Examples
    --------
    >>> from pgbm.sklearn import crps_ensemble, make_probabilistic_scorer
    >>> scorer = make_probabilistic_scorer(crps_ensemble, greater_is_better=False)
    >>> make_probabilistic_scorer
    make_scorer(crps_ensemble, greater_is_better=False)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from pgbm.sklearn import HistGradientBoostingRegressor
    >>> grid = GridSearchCV(HistGradientBoostingRegressor(), 
    ...                     param_grid={'learning_rate': [0.05, 0.1]},
    ...                     scoring=scorer)
    """

    sign = 1 if greater_is_better else -1
    return _ProbabilisticScorer(score_func, sign, kwargs)

@njit(float64(float64[::1, :], float64[::1]), 
      parallel=True, fastmath=True, nogil=True,
      cache=True)
def _crps_ensemble(yhat_dist, y):
    n_forecasts = yhat_dist.shape[0]
    n_samples = yhat_dist.shape[1]
    crps = 0.
    # Loop over the samples
    for sample in prange(n_samples):
        # Sort the forecasts in ascending order
        y_cdf = 0.
        yhat_cdf = 0.
        yhats_prev = 0.
        crps_sample = 0.
        ys = y[sample]
        # Loop over the forecasts per sample
        yhat_dist_c = np.ascontiguousarray(yhat_dist[:, sample])
        yhat_dist_csorted = np.sort(yhat_dist_c)
        for forecast in range(n_forecasts):
            yhats = yhat_dist_csorted[forecast]
            if y_cdf == 0. and ys < yhats:
                crps_sample += (
                ((ys - yhats_prev) * yhat_cdf**2)
                + ((yhats - ys) * (yhat_cdf - 1) ** 2)
                )
                y_cdf += 1
            else:
                crps_sample += (
                    (yhats - yhats_prev) * (yhat_cdf - y_cdf) ** 2
                )
            yhat_cdf += 1 / n_forecasts
            yhats_prev = yhats

        # In case y_cdf == 0 after the loop
        if y_cdf == 0.0:
            crps_sample += (ys - yhats)
        crps += (1 / n_samples) * crps_sample

    return crps

class _ProbabilisticScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        y_pred, y_std = method_caller(estimator, "predict", X, return_std=True)
        yhat_dist = method_caller(estimator, "sample", y_pred, y_std, n_estimates=1_000)

        return self._sign * self._score_func(yhat_dist, y_true)

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _normal(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            yhat[j, i] = np.random.normal(y[j], y_std[j])

    return yhat.T

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _studentt(y, y_std, n_estimates, seed, v=3):   
    n_samples = y.shape[0]
    factor = v / (v - 2)
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            scale = np.sqrt(y_std[j]**2 / factor)
            yhat[j, i] = (np.random.standard_t(v)
                        * scale + y[j])

    return yhat.T

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _laplace(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            scale = np.sqrt(0.5 * y_std[j]**2)
            yhat[j, i] = np.random.laplace(y[j], scale)

    return yhat.T    

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _logistic(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            scale = np.sqrt((3 * y_std[j]**2) / np.pi**2)
            yhat[j, i] = np.random.logistic(y[j], scale)

    return yhat.T  

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _lognormal(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            loc = np.log(y[j]**2 / np.sqrt(y_std[j]**2 + y[j]**2))
            scale = np.log(1 + y_std[j]**2 / y[j]**2)
            yhat_log = np.random.normal(loc, np.sqrt(scale))
            yhat[j, i] = np.exp(yhat_log)

    return yhat.T  

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _gumbel(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            scale = np.sqrt(6 * y_std[j]**2 / np.pi**2)
            loc = y[j] - scale * np.euler_gamma
            yhat[j, i] = np.random.gumbel(loc, scale)

    return yhat.T 

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _gamma(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            shape = y[j]**2 / y_std[j]
            scale = y[j] / shape
            yhat[j, i] = np.random.gamma(shape, scale)

    return yhat.T 

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _poisson(y, n_estimates, seed):   
    n_samples = y.shape[0]
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            yhat[j, i] = np.random.poisson(y[j])

    return yhat.T 

@njit(parallel=True, fastmath=True, nogil=True,
      cache=True)
def _negativebinomial(y, y_std, n_estimates, seed):   
    n_samples = y.shape[0]
    eps = 1.0e-15
    yhat = np.zeros((n_samples, n_estimates), dtype=Y_DTYPE)    
    for j in prange(n_samples):
        np.random.seed(seed + j)
        for i in range(n_estimates):
            p = np.minimum(np.maximum(y[j] * y_std[j]**2, eps), 1 - eps)
            n = (p * y[j]) / (1 - p)
            Y = np.random.gamma(n, (1.0 - p) / p)
            yhat[j, i] = np.random.poisson(Y)

    return yhat.T