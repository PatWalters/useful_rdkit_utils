import inspect
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array as _sk_check_array
from sklearn.utils.validation import check_is_fitted

# scikit-learn renamed `force_all_finite` -> `ensure_all_finite` in 1.6 and
# removed the old name in 1.8. Detect which one the installed version uses.
_FINITE_KW = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(_sk_check_array).parameters
    else "force_all_finite"
)


def _check_array_allow_nan(X, **kwargs):
    """Version-agnostic ``check_array`` that permits NaN but rejects inf."""
    kwargs[_FINITE_KW] = "allow-nan"
    return _sk_check_array(X, **kwargs)


class DescriptorPreprocessor(BaseEstimator, TransformerMixin):
    """Clean and standardize a molecular descriptor matrix.

    Performs the following steps, in order, on a numeric descriptor matrix:

    1. **Drop high-NaN columns** — any column whose fraction of NaN values
       exceeds ``max_nan_fraction`` is removed.
    2. **Drop constant columns** — any column whose variance (computed
       ignoring NaNs, only over columns that survived step 1) is less than
       or equal to ``variance_threshold`` is removed. This catches
       descriptors that are identical for every molecule and would carry no
       information.
    3. **Impute remaining NaNs** — leftover NaNs in the kept columns are
       filled with the per-column ``median`` or ``mean`` learned at fit time.
    4. **Standardize** — the surviving columns are mean-centered and scaled
       to unit variance with :class:`~sklearn.preprocessing.StandardScaler`
       (each step optional via ``with_mean`` / ``with_std``).

    The class follows scikit-learn conventions: hyperparameters are stored
    verbatim in ``__init__``, fitted state lives in trailing-underscore
    attributes, and the class inherits ``fit_transform`` from
    :class:`~sklearn.base.TransformerMixin`. It can therefore be dropped into
    a :class:`~sklearn.pipeline.Pipeline` or
    :class:`~sklearn.compose.ColumnTransformer`.

    Parameters
    ----------
    nan_strategy : {"median", "mean"}, default="median"
        Statistic used to impute remaining NaNs in columns that survive the
        ``max_nan_fraction`` filter. Median is more robust to outlier
        descriptor values.
    max_nan_fraction : float, default=0.2
        Columns whose fraction of NaN values strictly exceeds this threshold
        are dropped. Use 1.0 to keep every column regardless of NaN count,
        and 0.0 to drop any column containing a single NaN.
    variance_threshold : float, default=0.0
        Columns with variance ``<= variance_threshold`` (computed ignoring
        NaNs) are dropped. The default removes truly constant descriptors;
        raise it to also drop near-constant ones.
    with_mean : bool, default=True
        Mean-center each surviving column. Passed through to the underlying
        :class:`StandardScaler`.
    with_std : bool, default=True
        Scale each surviving column to unit variance. Passed through to the
        underlying :class:`StandardScaler`.

    Attributes
    ----------
    kept_columns_ : ndarray of bool, shape (n_features,)
        Boolean mask of columns retained after the NaN and variance filters.
        ``True`` means the column was kept.
    kept_indices_ : ndarray of int
        Integer indices of the retained columns, equivalent to
        ``np.where(kept_columns_)[0]``.
    impute_values_ : ndarray of shape (n_kept_features,)
        Per-column imputation value (median or mean) for the retained
        columns, computed on the training data ignoring NaNs.
    scaler_ : sklearn.preprocessing.StandardScaler
        Fitted scaler applied to the imputed, retained columns.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_features_out_ : int
        Number of features produced by ``transform``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0, 5.0, np.nan],
    ...               [2.0, 2.0, 6.0, 1.0],
    ...               [3.0, 2.0, 7.0, 2.0]])
    >>> pre = DescriptorPreprocessor()
    >>> X_clean = pre.fit_transform(X)
    >>> X_clean.shape  # constant column dropped, NaN imputed, rest scaled
    (3, 3)
    """

    def __init__(
        self,
        nan_strategy="median",
        max_nan_fraction=0.2,
        variance_threshold=0.0,
        with_mean=True,
        with_std=True,
    ):
        self.nan_strategy = nan_strategy
        self.max_nan_fraction = max_nan_fraction
        self.variance_threshold = variance_threshold
        self.with_mean = with_mean
        self.with_std = with_std

    def _validate_params(self):
        """Sanity-check hyperparameters before doing real work."""
        if self.nan_strategy not in ("median", "mean"):
            raise ValueError(
                f"nan_strategy must be 'median' or 'mean', got {self.nan_strategy!r}"
            )
        if not 0.0 <= self.max_nan_fraction <= 1.0:
            raise ValueError(
                f"max_nan_fraction must be in [0, 1], got {self.max_nan_fraction}"
            )
        if self.variance_threshold < 0:
            raise ValueError(
                f"variance_threshold must be >= 0, got {self.variance_threshold}"
            )

    def fit(self, X, y=None):
        """Learn the column mask, imputation values, and scaler.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw descriptor matrix. May contain NaNs (but not infs).
        y : Ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : DescriptorPreprocessor
            The fitted transformer.

        Raises
        ------
        ValueError
            If hyperparameters are invalid, or if every column gets dropped
            by the NaN and variance filters.
        """
        self._validate_params()
        X = _check_array_allow_nan(X, dtype=np.float64, ensure_2d=True)
        self.n_features_in_ = X.shape[1]

        # 1. NaN-fraction filter
        nan_fraction = np.isnan(X).mean(axis=0)
        nan_ok = nan_fraction <= self.max_nan_fraction

        # 2. Variance filter (ignoring NaNs). Compute variance only on
        #    columns that survived step 1, so we never hand `np.nanvar` an
        #    all-NaN slice and avoid the "Degrees of freedom <= 0" warning.
        variances = np.full(self.n_features_in_, np.nan)
        if nan_ok.any():
            variances[nan_ok] = np.nanvar(X[:, nan_ok], axis=0)
        var_ok = variances > self.variance_threshold

        self.kept_columns_ = nan_ok & var_ok
        self.kept_indices_ = np.where(self.kept_columns_)[0]

        if self.kept_indices_.size == 0:
            raise ValueError(
                "All columns were dropped by the NaN and/or variance filters. "
                "Loosen `max_nan_fraction` or lower `variance_threshold`."
            )

        X_kept = X[:, self.kept_columns_]

        # 3. Imputation values from training data. Surviving columns are
        #    guaranteed to have at least one non-NaN value (variance > 0
        #    implies >= 2 distinct finite values), so nanmedian / nanmean
        #    will not warn here. We still guard with a filter as a
        #    belt-and-braces measure.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            if self.nan_strategy == "median":
                self.impute_values_ = np.nanmedian(X_kept, axis=0)
            else:
                self.impute_values_ = np.nanmean(X_kept, axis=0)

        # Fill NaNs in the training matrix before fitting the scaler.
        X_imputed = self._impute(X_kept)

        # 4. Scaler
        self.scaler_ = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.scaler_.fit(X_imputed)

        self.n_features_out_ = self.kept_indices_.size
        return self

    def transform(self, X):
        """Apply the learned column mask, imputation, and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw descriptor matrix with the same number of columns as the
            matrix seen during ``fit``. May contain NaNs (but not infs).

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features_out_)
            Cleaned, imputed, and (optionally) standardized matrix.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit`` has not been called.
        ValueError
            If the number of features in ``X`` does not match the training
            matrix.
        """
        check_is_fitted(self, "scaler_")
        X = _check_array_allow_nan(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )
        X_kept = X[:, self.kept_columns_]
        X_imputed = self._impute(X_kept)
        return self.scaler_.transform(X_imputed)

    def _impute(self, X_kept):
        """Fill NaNs in ``X_kept`` with the per-column values from ``fit``.

        Parameters
        ----------
        X_kept : ndarray of shape (n_samples, n_kept_features)
            Matrix already restricted to the retained columns.

        Returns
        -------
        ndarray of the same shape, NaN-free.
        """
        X_out = X_kept.copy()
        mask = np.isnan(X_out)
        if mask.any():
            col_idx = np.where(mask)[1]
            X_out[mask] = self.impute_values_[col_idx]
        return X_out

    def get_feature_names_out(self, input_features=None):
        """Return the names of the retained features.

        Parameters
        ----------
        input_features : array-like of str, optional
            Names of the input columns. If None, generic names of the form
            ``"x0"``, ``"x1"``, ... are generated.

        Returns
        -------
        ndarray of str
            Names of the features produced by ``transform``, in order.
        """
        check_is_fitted(self, "kept_indices_")
        if input_features is None:
            input_features = np.array([f"x{i}" for i in range(self.n_features_in_)])
        else:
            input_features = np.asarray(input_features)
            if input_features.shape[0] != self.n_features_in_:
                raise ValueError(
                    f"input_features has length {input_features.shape[0]}, "
                    f"expected {self.n_features_in_}"
                )
        return input_features[self.kept_indices_]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    X[:, 2] = 3.14           # constant column -> dropped
    X[0, 0] = np.nan         # sporadic NaN  -> imputed
    X[:5, 4] = np.nan        # 25% NaN col, above default 0.2 threshold -> dropped

    pre = DescriptorPreprocessor()
    X_clean = pre.fit_transform(X)
    print(f"input shape  : {X.shape}")
    print(f"output shape : {X_clean.shape}")
    print(f"kept columns : {pre.kept_indices_}")
    print(f"col means    : {X_clean.mean(axis=0).round(3)}")  # ~0
    print(f"col stds     : {X_clean.std(axis=0).round(3)}")   # ~1
