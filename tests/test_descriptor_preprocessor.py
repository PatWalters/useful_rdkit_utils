import doctest

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import useful_rdkit_utils.descriptor_preprocessor as dp_module
from useful_rdkit_utils import DescriptorPreprocessor


@pytest.fixture
def descriptor_matrix():
    """Six columns exercising every filter: clean, constant, sparse NaN, all NaN."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    X[:, 2] = 3.14        # constant -> dropped by the variance filter
    X[0, 0] = np.nan      # 5% NaN -> kept and imputed
    X[:5, 4] = np.nan     # 25% NaN -> dropped by the NaN filter
    return X


def test_module_doctest_passes():
    """The class docstring carries a worked example; make sure it is actually run."""
    results = doctest.testmod(dp_module, verbose=False)
    assert results.attempted > 0
    assert results.failed == 0


def test_fit_transform_drops_and_scales(descriptor_matrix):
    pre = DescriptorPreprocessor()
    out = pre.fit_transform(descriptor_matrix)
    assert list(pre.kept_indices_) == [0, 1, 3, 5]
    assert out.shape == (20, 4)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1, atol=1e-9)
    assert not np.isnan(out).any()


def test_transform_applies_the_training_imputation(descriptor_matrix):
    pre = DescriptorPreprocessor().fit(descriptor_matrix)
    # a fresh row that is entirely NaN in the kept columns becomes the training medians,
    # which scale to approximately zero
    new = np.full((1, descriptor_matrix.shape[1]), np.nan)
    out = pre.transform(new)
    assert out.shape == (1, 4)
    assert not np.isnan(out).any()


def test_transform_rejects_a_different_column_count(descriptor_matrix):
    pre = DescriptorPreprocessor().fit(descriptor_matrix)
    with pytest.raises(ValueError, match="Expected 6 features"):
        pre.transform(np.zeros((3, 5)))


def test_transform_before_fit_raises(descriptor_matrix):
    with pytest.raises(NotFittedError):
        DescriptorPreprocessor().transform(descriptor_matrix)


def test_mean_strategy(descriptor_matrix):
    pre = DescriptorPreprocessor(nan_strategy="mean").fit(descriptor_matrix)
    kept = descriptor_matrix[:, pre.kept_columns_]
    assert np.allclose(pre.impute_values_, np.nanmean(kept, axis=0))


def test_median_strategy(descriptor_matrix):
    pre = DescriptorPreprocessor(nan_strategy="median").fit(descriptor_matrix)
    kept = descriptor_matrix[:, pre.kept_columns_]
    assert np.allclose(pre.impute_values_, np.nanmedian(kept, axis=0))


@pytest.mark.parametrize("kwargs,message", [
    ({"nan_strategy": "mode"}, "nan_strategy"),
    ({"max_nan_fraction": 1.5}, "max_nan_fraction"),
    ({"max_nan_fraction": -0.1}, "max_nan_fraction"),
    ({"variance_threshold": -1}, "variance_threshold"),
])
def test_invalid_hyperparameters(descriptor_matrix, kwargs, message):
    with pytest.raises(ValueError, match=message):
        DescriptorPreprocessor(**kwargs).fit(descriptor_matrix)


def test_all_columns_dropped_raises():
    X = np.full((5, 3), np.nan)
    with pytest.raises(ValueError, match="All columns were dropped"):
        DescriptorPreprocessor().fit(X)


def test_max_nan_fraction_of_one_keeps_sparse_columns(descriptor_matrix):
    pre = DescriptorPreprocessor(max_nan_fraction=1.0).fit(descriptor_matrix)
    # only the constant column goes now
    assert list(pre.kept_indices_) == [0, 1, 3, 4, 5]


def test_variance_threshold_drops_near_constant_columns():
    X = np.column_stack([np.arange(10.0), np.full(10, 5.0) + np.arange(10) * 1e-4])
    pre = DescriptorPreprocessor(variance_threshold=1e-3).fit(X)
    assert list(pre.kept_indices_) == [0]


def test_get_feature_names_out(descriptor_matrix):
    pre = DescriptorPreprocessor().fit(descriptor_matrix)
    assert list(pre.get_feature_names_out()) == ["x0", "x1", "x3", "x5"]
    names = ["MolWt", "LogP", "Const", "TPSA", "Sparse", "HBD"]
    assert list(pre.get_feature_names_out(names)) == ["MolWt", "LogP", "TPSA", "HBD"]


def test_get_feature_names_out_checks_length(descriptor_matrix):
    pre = DescriptorPreprocessor().fit(descriptor_matrix)
    with pytest.raises(ValueError, match="input_features has length"):
        pre.get_feature_names_out(["a", "b"])


def test_infinite_values_are_rejected(descriptor_matrix):
    descriptor_matrix[1, 1] = np.inf
    with pytest.raises(ValueError):
        DescriptorPreprocessor().fit(descriptor_matrix)


def test_works_inside_a_sklearn_pipeline(descriptor_matrix):
    y = np.arange(descriptor_matrix.shape[0], dtype=float)
    pipe = Pipeline([("pre", DescriptorPreprocessor()), ("model", LinearRegression())])
    pipe.fit(descriptor_matrix, y)
    assert pipe.predict(descriptor_matrix).shape == (20,)


def test_with_mean_and_with_std_are_honoured(descriptor_matrix):
    out = DescriptorPreprocessor(with_mean=False, with_std=False).fit_transform(descriptor_matrix)
    kept = descriptor_matrix[:, [0, 1, 3, 5]]
    expected = np.where(np.isnan(kept), np.nanmedian(kept, axis=0), kept)
    assert np.allclose(out, expected)
