import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn._loss.loss import (
    AbsoluteError,
    HalfBinomialLoss,
    HalfSquaredError,
    PinballLoss,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import make_low_rank_matrix
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.base import is_regressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_poisson_deviance
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.compose import make_column_transformer

from pgbm.sklearn import HistGradientBoostingRegressor
from pgbm.sklearn.grower import TreeGrower
from pgbm.sklearn.binning import _BinMapper
from pgbm.sklearn.common import G_H_DTYPE
from sklearn.utils import shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads


n_threads = _openmp_effective_n_threads()

X_classification, y_classification = make_classification(random_state=0)
X_regression, y_regression = make_regression(random_state=0)
X_multi_classification, y_multi_classification = make_classification(
    n_classes=3, n_informative=3, random_state=0
)


def _make_dumb_dataset(n_samples):
    """Make a dumb dataset to test early stopping."""
    rng = np.random.RandomState(42)
    X_dumb = rng.randn(n_samples, 1)
    y_dumb = (X_dumb[:, 0] > 0).astype("int64")
    return X_dumb, y_dumb


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
@pytest.mark.parametrize(
    "params, err_msg",
    [
        (
            {"interaction_cst": "string"},
            "",
        ),
        (
            {"interaction_cst": [0, 1]},
            "Interaction constraints must be None or an iterable of iterables",
        ),
        (
            {"interaction_cst": [{0, 9999}]},
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
        (
            {"interaction_cst": [{-1, 0}]},
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
        (
            {"interaction_cst": [{0.5}]},
            r"Interaction constraints must consist of integer indices in \[0,"
            r" n_features - 1\] = \[.*\], specifying the position of features,",
        ),
    ],
)
def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):

    with pytest.raises(ValueError, match=err_msg):
        GradientBoosting(**params).fit(X, y)


@pytest.mark.parametrize(
    "scoring, validation_fraction, early_stopping, n_iter_no_change, tol",
    [
        ("neg_mean_squared_error", 0.1, True, 5, 1e-7),  # use scorer
        ("neg_mean_squared_error", None, True, 5, 1e-1),  # use scorer on train
        (None, 0.1, True, 5, 1e-7),  # same with default scorer
        (None, None, True, 5, 1e-1),
        ("loss", 0.1, True, 5, 1e-7),  # use loss
        ("loss", None, True, 5, 1e-1),  # use loss on training data
        (None, None, False, 5, 0.0),  # no early stopping
    ],
)
def test_early_stopping_regression(
    scoring, validation_fraction, early_stopping, n_iter_no_change, tol
):

    max_iter = 200

    X, y = make_regression(n_samples=50, random_state=0)

    gb = HistGradientBoostingRegressor(
        verbose=1,  # just for coverage
        min_samples_leaf=5,  # easier to overfit fast
        scoring=scoring,
        tol=tol,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        random_state=0,
    )
    gb.fit(X, y)

    if early_stopping:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter


@pytest.mark.parametrize(
    "GradientBoosting, X, y",
    [
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10000)),
        (HistGradientBoostingRegressor, *_make_dumb_dataset(10001)),
    ],
)
def test_early_stopping_default(GradientBoosting, X, y):
    # Test that early stopping is enabled by default if and only if there
    # are more than 10000 samples
    gb = GradientBoosting(max_iter=10, n_iter_no_change=2, tol=1e-1)
    gb.fit(X, y)
    if X.shape[0] > 10000:
        assert gb.n_iter_ < gb.max_iter
    else:
        assert gb.n_iter_ == gb.max_iter


def test_absolute_error():
    # For coverage only.
    X, y = make_regression(n_samples=500, random_state=0)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error", random_state=0)
    gbdt.fit(X, y)
    assert gbdt.score(X, y) > 0.9


def test_absolute_error_sample_weight():
    # non regression test for issue #19400
    # make sure no error is thrown during fit of
    # HistGradientBoostingRegressor with absolute_error loss function
    # and passing sample_weight
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    y = rng.uniform(-1, 1, size=n_samples)
    sample_weight = rng.uniform(0, 1, size=n_samples)
    gbdt = HistGradientBoostingRegressor(loss="absolute_error")
    gbdt.fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("quantile", [0.2, 0.5, 0.8])
def test_asymmetric_error(quantile):
    """Test quantile regression for asymmetric distributed targets."""
    n_samples = 10_000
    rng = np.random.RandomState(42)
    # take care that X @ coef + intercept > 0
    X = np.concatenate(
        (
            np.abs(rng.randn(n_samples)[:, None]),
            -rng.randint(2, size=(n_samples, 1)),
        ),
        axis=1,
    )
    intercept = 1.23
    coef = np.array([0.5, -2])
    # For an exponential distribution with rate lambda, e.g. exp(-lambda * x),
    # the quantile at level q is:
    #   quantile(q) = - log(1 - q) / lambda
    #   scale = 1/lambda = -quantile(q) / log(1-q)
    y = rng.exponential(
        scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples
    )
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=25,
        random_state=0,
        max_leaf_nodes=10,
    ).fit(X, y)
    assert_allclose(np.mean(model.predict(X) > y), quantile, rtol=1e-2)

    pinball_loss = PinballLoss(quantile=quantile)
    loss_true_quantile = pinball_loss(y, X @ coef + intercept)
    loss_pred_quantile = pinball_loss(y, model.predict(X))
    # we are overfitting
    assert loss_pred_quantile <= loss_true_quantile


@pytest.mark.parametrize("y", [([1.0, -2.0, 0.0]), ([0.0, 0.0, 0.0])])
def test_poisson_y_positive(y):
    # Test that ValueError is raised if either one y_i < 0 or sum(y_i) <= 0.
    err_msg = r"loss='poisson' requires non-negative y and sum\(y\) > 0."
    gbdt = HistGradientBoostingRegressor(loss="poisson", random_state=0)
    with pytest.raises(ValueError, match=err_msg):
        gbdt.fit(np.zeros(shape=(len(y), 1)), y)


def test_poisson():
    # For Poisson distributed target, Poisson loss should give better results
    # than least squares measured in Poisson deviance as metric.
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = 500, 100, 100
    X = make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # We create a log-linear Poisson model and downscale coef as it will get
    # exponentiated.
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    gbdt_pois = HistGradientBoostingRegressor(
        loss="poisson", random_state=rng, with_variance=True, distribution="poisson"
    )
    gbdt_ls = HistGradientBoostingRegressor(loss="squared_error", random_state=rng)
    gbdt_pois.fit(X_train, y_train)
    gbdt_ls.fit(X_train, y_train)
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)

    for X, y in [(X_train, y_train), (X_test, y_test)]:
        yhat_mean_pois, yhat_std_pois = gbdt_pois.predict(X, return_std=True)
        yhat_dist_pois = gbdt_pois.sample(
            yhat_mean_pois, yhat_std_pois, n_estimates=10_000
        )
        yhat_mean_pois_sampled = yhat_dist_pois.mean(axis=0)
        metric_pois = mean_poisson_deviance(y, yhat_mean_pois)
        metric_pois_sampled = mean_poisson_deviance(y, yhat_mean_pois_sampled)
        # squared_error might produce non-positive predictions => clip
        metric_ls = mean_poisson_deviance(y, np.clip(gbdt_ls.predict(X), 1e-15, None))
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        assert metric_pois < metric_ls
        assert metric_pois < metric_dummy
        assert metric_pois_sampled < metric_ls
        assert metric_pois_sampled < metric_dummy


@pytest.mark.parametrize(
    "missing_proportion, expected_min_score_classification, "
    "expected_min_score_regression",
    [(0.1, 0.97, 0.89), (0.2, 0.93, 0.81), (0.5, 0.79, 0.52)],
)
def test_missing_values_resilience(
    missing_proportion,
    expected_min_score_classification,
    expected_min_score_regression,
):
    # Make sure the estimators can deal with missing values and still yield
    # decent predictions

    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        random_state=rng,
    )
    gb = HistGradientBoostingRegressor()
    expected_min_score = expected_min_score_regression

    mask = rng.binomial(1, missing_proportion, size=X.shape).astype(bool)
    X[mask] = np.nan

    gb.fit(X, y)

    assert gb.score(X, y) > expected_min_score

def test_missing_values_minmax_imputation():
    # Compare the buit-in missing value handling of Histogram GBC with an
    # a-priori missing value imputation strategy that should yield the same
    # results in terms of decision function.
    #
    # Each feature (containing NaNs) is replaced by 2 features:
    # - one where the nans are replaced by min(feature) - 1
    # - one where the nans are replaced by max(feature) + 1
    # A split where nans go to the left has an equivalent split in the
    # first (min) feature, and a split where nans go to the right has an
    # equivalent split in the second (max) feature.
    #
    # Assuming the data is such that there is never a tie to select the best
    # feature to split on during training, the learned decision trees should be
    # strictly equivalent (learn a sequence of splits that encode the same
    # decision function).
    #
    # The MinMaxImputer transformer is meant to be a toy implementation of the
    # "Missing In Attributes" (MIA) missing value handling for decision trees
    # https://www.sciencedirect.com/science/article/abs/pii/S0167865508000305
    # The implementation of MIA as an imputation transformer was suggested by
    # "Remark 3" in :arxiv:'<1902.06931>`

    class MinMaxImputer(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            mm = MinMaxScaler().fit(X)
            self.data_min_ = mm.data_min_
            self.data_max_ = mm.data_max_
            return self

        def transform(self, X):
            X_min, X_max = X.copy(), X.copy()

            for feature_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, feature_idx])
                X_min[nan_mask, feature_idx] = self.data_min_[feature_idx] - 1
                X_max[nan_mask, feature_idx] = self.data_max_[feature_idx] + 1

            return np.concatenate([X_min, X_max], axis=1)

    def make_missing_value_data(n_samples=int(1e4), seed=0):
        rng = np.random.RandomState(seed)
        X, y = make_regression(n_samples=n_samples, n_features=4, random_state=rng)

        # Pre-bin the data to ensure a deterministic handling by the 2
        # strategies and also make it easier to insert np.nan in a structured
        # way:
        X = KBinsDiscretizer(n_bins=42, encode="ordinal").fit_transform(X)

        # First feature has missing values completely at random:
        rnd_mask = rng.rand(X.shape[0]) > 0.9
        X[rnd_mask, 0] = np.nan

        # Second and third features have missing values for extreme values
        # (censoring missingness):
        low_mask = X[:, 1] == 0
        X[low_mask, 1] = np.nan

        high_mask = X[:, 2] == X[:, 2].max()
        X[high_mask, 2] = np.nan

        # Make the last feature nan pattern very informative:
        y_max = np.percentile(y, 70)
        y_max_mask = y >= y_max
        y[y_max_mask] = y_max
        X[y_max_mask, 3] = np.nan

        # Check that there is at least one missing value in each feature:
        for feature_idx in range(X.shape[1]):
            assert any(np.isnan(X[:, feature_idx]))

        # Let's use a test set to check that the learned decision function is
        # the same as evaluated on unseen data. Otherwise it could just be the
        # case that we find two independent ways to overfit the training set.
        return train_test_split(X, y, random_state=rng)

    # n_samples need to be large enough to minimize the likelihood of having
    # several candidate splits with the same gain value in a given tree.
    X_train, X_test, y_train, y_test = make_missing_value_data(
        n_samples=int(1e4), seed=0
    )

    # Use a small number of leaf nodes and iterations so as to keep
    # under-fitting models to minimize the likelihood of ties when training the
    # model.
    gbm1 = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=5, random_state=0)
    gbm1.fit(X_train, y_train)

    gbm2 = make_pipeline(MinMaxImputer(), clone(gbm1))
    gbm2.fit(X_train, y_train)

    # Check that the model reach the same score:
    assert gbm1.score(X_train, y_train) == pytest.approx(gbm2.score(X_train, y_train))

    assert gbm1.score(X_test, y_test) == pytest.approx(gbm2.score(X_test, y_test))

    # Check the individual prediction match as a finer grained
    # decision function check.
    assert_allclose(gbm1.predict(X_train), gbm2.predict(X_train))
    assert_allclose(gbm1.predict(X_test), gbm2.predict(X_test))


def test_infinite_values():
    # Basic test for infinite values

    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])

    gbdt = HistGradientBoostingRegressor(min_samples_leaf=1)
    gbdt.fit(X, y)
    np.testing.assert_allclose(gbdt.predict(X), y, atol=1e-4)


def test_consistent_lengths():
    X = np.array([-np.inf, 0, 1, np.inf]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    sample_weight = np.array([0.1, 0.3, 0.1])
    gbdt = HistGradientBoostingRegressor()
    with pytest.raises(ValueError, match=r"sample_weight.shape == \(3,\), expected"):
        gbdt.fit(X, y, sample_weight)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent number"
    ):
        gbdt.fit(X, y[1:])


def test_zero_sample_weights_regression():
    # Make sure setting a SW to zero amounts to ignoring the corresponding
    # sample

    X = [[1, 0], [1, 0], [1, 0], [0, 1]]
    y = [0, 0, 1, 0]
    # ignore the first 2 training samples by setting their weight to 0
    sample_weight = [0, 0, 1, 1]
    gb = HistGradientBoostingRegressor(min_samples_leaf=1)
    gb.fit(X, y, sample_weight=sample_weight)
    assert gb.predict([[1, 0]])[0] > 0.5

@pytest.mark.parametrize("duplication", ("half", "all"))
def test_sample_weight_effect(duplication):
    # High level test to make sure that duplicating a sample is equivalent to
    # giving it weight of 2.

    # fails for n_samples > 255 because binning does not take sample weights
    # into account. Keeping n_samples <= 255 makes
    # sure only unique values are used so SW have no effect on binning.
    n_samples = 255
    n_features = 2
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        random_state=0,
    )
    Klass = HistGradientBoostingRegressor

    # This test can't pass if min_samples_leaf > 1 because that would force 2
    # samples to be in the same node in est_sw, while these samples would be
    # free to be separate in est_dup: est_dup would just group together the
    # duplicated samples.
    est = Klass(min_samples_leaf=1)

    # Create dataset with duplicate and corresponding sample weights
    if duplication == "half":
        lim = n_samples // 2
    else:
        lim = n_samples
    X_dup = np.r_[X, X[:lim]]
    y_dup = np.r_[y, y[:lim]]
    sample_weight = np.ones(shape=(n_samples))
    sample_weight[:lim] = 2

    est_sw = clone(est).fit(X, y, sample_weight=sample_weight)
    est_dup = clone(est).fit(X_dup, y_dup)

    # checking raw_predict is stricter than just predict for classification
    assert np.allclose(est_sw._raw_predict(X_dup), est_dup._raw_predict(X_dup))


@pytest.mark.parametrize("Loss", (HalfSquaredError, AbsoluteError))
def test_sum_hessians_are_sample_weight(Loss):
    # For losses with constant hessians, the sum_hessians field of the
    # histograms must be equal to the sum of the sample weight of samples at
    # the corresponding bin.

    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)

    # While sample weights are supposed to be positive, this still works.
    sample_weight = rng.normal(size=n_samples)

    loss = Loss(sample_weight=sample_weight)
    gradients, hessians = loss.init_gradient_and_hessian(
        n_samples=n_samples, dtype=G_H_DTYPE
    )
    gradients, hessians = gradients.reshape((-1, 1)), hessians.reshape((-1, 1))
    raw_predictions = rng.normal(size=(n_samples, 1))
    loss.gradient_hessian(
        y_true=y,
        raw_prediction=raw_predictions,
        sample_weight=sample_weight,
        gradient_out=gradients,
        hessian_out=hessians,
        n_threads=n_threads,
    )

    # build sum_sample_weight which contains the sum of the sample weights at
    # each bin (for each feature). This must be equal to the sum_hessians
    # field of the corresponding histogram
    sum_sw = np.zeros(shape=(n_features, bin_mapper.n_bins))
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            sum_sw[feature_idx, X_binned[sample_idx, feature_idx]] += sample_weight[
                sample_idx
            ]

    # Build histogram
    grower = TreeGrower(
        X_binned, gradients[:, 0], hessians[:, 0], n_bins=bin_mapper.n_bins
    )
    histograms = grower.histogram_builder.compute_histograms_brute(
        grower.root.sample_indices
    )

    for feature_idx in range(n_features):
        for bin_idx in range(bin_mapper.n_bins):
            assert histograms[feature_idx, bin_idx]["sum_hessians"] == (
                pytest.approx(sum_sw[feature_idx, bin_idx], rel=1e-5)
            )


def test_single_node_trees():
    # Make sure it's still possible to build single-node trees. In that case
    # the value of the root is set to 0. That's a correct value: if the tree is
    # single-node that's because min_gain_to_split is not respected right from
    # the root, so we don't want the tree to have any impact on the
    # predictions.

    X, y = make_classification(random_state=0)
    y[:] = 1  # constant target will lead to a single root node

    est = HistGradientBoostingRegressor(max_iter=20)
    est.fit(X, y)

    assert all(len(predictor[0].nodes) == 1 for predictor in est._predictors)
    assert all(predictor[0].nodes[0]["value"] == 0 for predictor in est._predictors)
    # Still gives correct predictions thanks to the baseline prediction
    assert_allclose(est.predict(X), y)


@pytest.mark.parametrize(
    "Est, loss, X, y",
    [
        (
            HistGradientBoostingRegressor,
            HalfSquaredError(sample_weight=None),
            X_regression,
            y_regression,
        ),
    ],
)
def test_custom_loss(Est, loss, X, y):
    est = Est(loss=loss, max_iter=20)
    est.fit(X, y)


@pytest.mark.parametrize(
    "HistGradientBoosting, X, y",
    [
        (HistGradientBoostingRegressor, X_regression, y_regression),
    ],
)
def test_staged_predict(HistGradientBoosting, X, y):

    # Test whether staged predictor eventually gives
    # the same prediction.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )
    gb = HistGradientBoosting(max_iter=10)

    # test raise NotFittedError if not fitted
    with pytest.raises(NotFittedError):
        next(gb.staged_predict(X_test))

    gb.fit(X_train, y_train)

    # test if the staged predictions of each iteration
    # are equal to the corresponding predictions of the same estimator
    # trained from scratch.
    # this also test limit case when max_iter = 1
    method_names = (
        ["predict"]
        if is_regressor(gb)
        else ["predict", "predict_proba", "decision_function"]
    )
    for method_name in method_names:

        staged_method = getattr(gb, "staged_" + method_name)
        staged_predictions = list(staged_method(X_test))
        assert len(staged_predictions) == gb.n_iter_
        for n_iter, staged_predictions in enumerate(staged_method(X_test), 1):
            aux = HistGradientBoosting(max_iter=n_iter)
            aux.fit(X_train, y_train)
            pred_aux = getattr(aux, method_name)(X_test)

            assert_allclose(staged_predictions, pred_aux)
            assert staged_predictions.shape == pred_aux.shape


@pytest.mark.parametrize("insert_missing", [False, True])
@pytest.mark.parametrize("bool_categorical_parameter", [True, False])
def test_unknown_categories_nan(insert_missing, bool_categorical_parameter):
    # Make sure no error is raised at predict if a category wasn't seen during
    # fit. We also make sure they're treated as nans.

    rng = np.random.RandomState(0)
    n_samples = 1000
    f1 = rng.rand(n_samples)
    f2 = rng.randint(4, size=n_samples)
    X = np.c_[f1, f2]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    if bool_categorical_parameter:
        categorical_features = [False, True]
    else:
        categorical_features = [1]

    if insert_missing:
        mask = rng.binomial(1, 0.01, size=X.shape).astype(bool)
        assert mask.sum() > 0
        X[mask] = np.nan

    est = HistGradientBoostingRegressor(max_iter=20, categorical_features=categorical_features).fit(X, y)
    assert_array_equal(est.is_categorical_, [False, True])

    # Make sure no error is raised on unknown categories and nans
    # unknown categories will be treated as nans
    X_test = np.zeros((10, X.shape[1]), dtype=float)
    X_test[:5, 1] = 30
    X_test[5:, 1] = np.nan
    assert len(np.unique(est.predict(X_test))) == 1


@pytest.mark.parametrize(
    "categorical_features, monotonic_cst, expected_msg",
    [
        (
            ["hello", "world"],
            None,
            "categorical_features must be an array-like of bools or array-like of "
            "ints.",
        ),
        (
            [0, -1],
            None,
            (
                r"categorical_features set as integer indices must be in "
                r"\[0, n_features - 1\]"
            ),
        ),
        (
            [True, True, False, False, True],
            None,
            r"categorical_features set as a boolean mask must have shape "
            r"\(n_features,\)",
        ),
        (
            [True, True, False, False],
            [0, -1, 0, 1],
            "Categorical features cannot have monotonic constraints",
        ),
    ],
)
def test_categorical_spec_errors(
    categorical_features, monotonic_cst, expected_msg
):
    # Test errors when categories are specified incorrectly
    n_samples = 100
    X, y = make_classification(random_state=0, n_features=4, n_samples=n_samples)
    rng = np.random.RandomState(0)
    X[:, 0] = rng.randint(0, 10, size=n_samples)
    X[:, 1] = rng.randint(0, 10, size=n_samples)
    est = HistGradientBoostingRegressor(categorical_features=categorical_features, monotonic_cst=monotonic_cst)

    with pytest.raises(ValueError, match=expected_msg):
        est.fit(X, y)


@pytest.mark.parametrize("categorical_features", ([False, False], []))
@pytest.mark.parametrize("as_array", (True, False))
def test_categorical_spec_no_categories(categorical_features, as_array):
    # Make sure we can properly detect that no categorical features are present
    # even if the categorical_features parameter is not None
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)
    if as_array:
        categorical_features = np.asarray(categorical_features)
    est = HistGradientBoostingRegressor(categorical_features=categorical_features).fit(X, y)
    assert est.is_categorical_ is None


def test_categorical_bad_encoding_errors():
    # Test errors when categories are encoded incorrectly

    gb = HistGradientBoostingRegressor(categorical_features=[True], max_bins=2)

    X = np.array([[0, 1, 2]]).T
    y = np.arange(3)
    msg = "Categorical feature at index 0 is expected to have a cardinality <= 2"
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)

    X = np.array([[0, 2]]).T
    y = np.arange(2)
    msg = "Categorical feature at index 0 is expected to be encoded with values < 2"
    with pytest.raises(ValueError, match=msg):
        gb.fit(X, y)

    # nans are ignored in the counts
    X = np.array([[0, 1, np.nan]]).T
    y = np.arange(3)
    gb.fit(X, y)


def test_uint8_predict():
    # Non regression test for
    # https://github.com/scikit-learn/scikit-learn/issues/18408
    # Make sure X can be of dtype uint8 (i.e. X_BINNED_DTYPE) in predict. It
    # will be converted to X_DTYPE.

    rng = np.random.RandomState(0)

    X = rng.randint(0, 100, size=(10, 2)).astype(np.uint8)
    y = rng.randint(0, 2, size=10).astype(np.uint8)
    est = HistGradientBoostingRegressor()
    est.fit(X, y)
    est.predict(X)


@pytest.mark.parametrize(
    "interaction_cst, n_features, result",
    [
        (None, 931, None),
        ([{0, 1}], 2, [{0, 1}]),
        ([(1, 0), [5, 1]], 6, [{0, 1}, {1, 5}, {2, 3, 4}]),
    ],
)
def test_check_interaction_cst(interaction_cst, n_features, result):
    """Check that _check_interaction_cst returns the expected list of sets"""
    est = HistGradientBoostingRegressor()
    est.set_params(interaction_cst=interaction_cst)
    assert est._check_interaction_cst(n_features) == result


def test_interaction_cst_numerically():
    """Check that interaction constraints have no forbidden interactions."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = rng.uniform(size=(n_samples, 2))
    # Construct y with a strong interaction term
    # y = x0 + x1 + 5 * x0 * x1
    y = np.hstack((X, 5 * X[:, [0]] * X[:, [1]])).sum(axis=1)

    est = HistGradientBoostingRegressor(random_state=42)
    est.fit(X, y)
    est_no_interactions = HistGradientBoostingRegressor(
        interaction_cst=[{0}, {1}], random_state=42
    )
    est_no_interactions.fit(X, y)

    delta = 0.25
    # Make sure we do not extrapolate out of the training set as tree-based estimators
    # are very bad in doing so.
    X_test = X[(X[:, 0] < 1 - delta) & (X[:, 1] < 1 - delta)]
    X_delta_d_0 = X_test + [delta, 0]
    X_delta_0_d = X_test + [0, delta]
    X_delta_d_d = X_test + [delta, delta]

    # Note: For the y from above as a function of x0 and x1, we have
    # y(x0+d, x1+d) = y(x0, x1) + 5 * d * (2/5 + x0 + x1) + 5 * d**2
    # y(x0+d, x1)   = y(x0, x1) + 5 * d * (1/5 + x1)
    # y(x0,   x1+d) = y(x0, x1) + 5 * d * (1/5 + x0)
    # Without interaction constraints, we would expect a result of 5 * d**2 for the
    # following expression, but zero with constraints in place.
    assert_allclose(
        est_no_interactions.predict(X_delta_d_d)
        + est_no_interactions.predict(X_test)
        - est_no_interactions.predict(X_delta_d_0)
        - est_no_interactions.predict(X_delta_0_d),
        0,
        atol=1e-12,
    )

    # Correct result of the expressions is 5 * delta**2. But this is hard to achieve by
    # a fitted tree-based model. However, with 100 iterations the expression should
    # at least be positive!
    assert np.all(
        est.predict(X_delta_d_d)
        + est.predict(X_test)
        - est.predict(X_delta_d_0)
        - est.predict(X_delta_0_d)
        > 0.01
    )


def test_no_user_warning_with_scoring():
    """Check that no UserWarning is raised when scoring is set.

    Non-regression test for #22907.
    """
    pd = pytest.importorskip("pandas")
    X, y = make_regression(n_samples=50, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])

    est = HistGradientBoostingRegressor(
        random_state=0, scoring="neg_mean_absolute_error", early_stopping=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        est.fit(X_df, y)


def test_unknown_category_that_are_negative():
    """Check that unknown categories that are negative does not error.

    Non-regression test for #24274.
    """
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = np.c_[rng.rand(n_samples), rng.randint(4, size=n_samples)]
    y = np.zeros(shape=n_samples)
    y[X[:, 1] % 2 == 0] = 1

    hist = HistGradientBoostingRegressor(
        random_state=0,
        categorical_features=[False, True],
        max_iter=10,
    ).fit(X, y)

    # Check that negative values from the second column are treated like a
    # missing category
    X_test_neg = np.asarray([[1, -2], [3, -4]])
    X_test_nan = np.asarray([[1, np.nan], [3, np.nan]])

    assert_allclose(hist.predict(X_test_neg), hist.predict(X_test_nan))


def test_distribution_means_and_variances():
    """Check that the sampled mean and variances from the output of a distribution
    are approximately equal to the empirical mean and variances from the model.

    In other words, this tests whether we correctly convert the empirical mean
    and variance from our model to appropriate parameters to parameterize a
    distribution.

    We test it by verifying that the difference between the sampled mean/variance
    and the true mean/variance is reducing if the number of sampled predictions
    increases. Note that this is not perfect, we only establish a reduction of
    errors. For some distributions we could also use an inequality
    (e.g. Hoeffdings) to check if the error makes sense.
    """
    rng = np.random.RandomState(42)
    # We use the same synthetic data as with the Poisson test
    n_train, n_test, n_features = 500, 100, 100
    X = make_low_rank_matrix(
        n_samples=n_train + n_test, n_features=n_features, random_state=rng
    )
    # We create a log-linear Poisson model and downscale coef as it will get
    # exponentiated.
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=rng
    )
    gbdt = HistGradientBoostingRegressor(loss="squared_error", with_variance=True)
    gbdt.fit(X_train, y_train)
    samples_array = [1, 10, 100, 1_000, 10_000, 100_000]
    distributions = [
        "normal",
        "studentt",
        "laplace",
        "logistic",
        "lognormal",
        "gamma",
        "gumbel",
        "poisson",
        "negativebinomial",
    ]
    for distribution in distributions:
        error = np.zeros((len(samples_array), len(X_test)))
        params = {"distribution":distribution}
        gbdt.set_params(**params)
        yhat_mean, yhat_std = gbdt.predict(X_test, return_std=True)
        yhat_mean = np.clip(yhat_mean, 1e-15, None)
        for i, samples in enumerate(samples_array):
            yhat_dist = gbdt.sample(yhat_mean, yhat_std, n_estimates=samples)
            yhat_mean_sampled = yhat_dist.mean(axis=0)
            error[i] = np.abs(yhat_mean_sampled - yhat_mean)
        print(distribution)
        assert np.all(np.diff(error.mean(axis=1)) < 0)