"""Utilities for comparing the performance of several ML models on the same dataset.

The module provides:

* :class:`WrapperFactory` - builds lightweight model wrapper classes that share a
  common ``fit`` / ``predict`` / ``validate`` interface backed by a precomputed
  SMILES-to-descriptor dictionary.
* :func:`get_performance_stats` - aggregates per-fold :math:`R^2` and MAE values
  for a set of methods evaluated on the same cross-validation splits.
* :func:`make_tukey_plot` - draws a Tukey HSD "simultaneous" plot to visualize
  pairwise differences between methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import useful_rdkit_utils as uru


class WrapperFactory:
    """Factory for building model wrapper classes with a uniform interface.

    The generated wrappers all expose ``fit(train)``, ``predict(test)``, and
    ``validate(train, test)`` methods that read SMILES from a ``SMILES`` column
    and look up precomputed descriptors in a shared dictionary. This lets a
    benchmark loop treat heterogeneous scikit-learn-style models uniformly.
    """

    @staticmethod
    def create_wrapper_class(model_class, descriptor_dict, class_name="DynamicMLWrapper", **model_kwargs):
        """Dynamically create a wrapper class around ``model_class``.

        Any extra keyword arguments are forwarded to the underlying model
        constructor each time the wrapper is instantiated.

        :param model_class: A scikit-learn-style estimator class implementing
            ``fit`` and ``predict``.
        :param descriptor_dict: Mapping from SMILES string to a fixed-length
            descriptor / fingerprint vector (anything ``np.stack`` can stack).
        :param class_name: Name assigned to the dynamically created class.
        :param model_kwargs: Keyword arguments forwarded to ``model_class``
            when the wrapper is instantiated.
        :return: A new class whose instances accept ``y_col`` (the target
            column name) and expose ``fit``, ``predict``, and ``validate``.
        """

        def __init__(self, y_col):
            """Instantiate the wrapped model and remember the target column."""
            self.model = model_class(**model_kwargs)
            self.y_col = y_col
            self.fp_name = "fp"
            self.descriptor_dict = descriptor_dict

        def fit(self, train):
            """Fit the wrapped model on ``train`` using ``descriptor_dict``.

            :param train: DataFrame with a ``SMILES`` column and ``self.y_col``.
            """
            train[self.fp_name] = train.SMILES.map(self.descriptor_dict)
            self.model.fit(np.stack(train[self.fp_name]), train[self.y_col])

        def predict(self, test):
            """Predict ``self.y_col`` for the rows in ``test``.

            :param test: DataFrame with a ``SMILES`` column.
            :return: 1D array of predictions aligned with ``test``.
            """
            test[self.fp_name] = test.SMILES.map(self.descriptor_dict)
            pred = self.model.predict(np.stack(test[self.fp_name]))
            return pred

        def validate(self, train, test):
            """Fit on ``train`` then predict ``test`` in one call.

            :param train: Training DataFrame (see :meth:`fit`).
            :param test: Test DataFrame (see :meth:`predict`).
            :return: Predictions for ``test``.
            """
            self.fit(train)
            return self.predict(test)

        class_attributes = {
            "__init__": __init__,
            "fit": fit,
            "predict": predict,
            "validate": validate
        }

        return type(class_name, (object,), class_attributes)

def get_performance_stats(df: pd.DataFrame, y_col: str, method_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate R^2 and MAE statistics for each method in the method list.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param y_col: The name of the column containing the true values.
    :type y_col: str
    :param method_list: A list of method names to evaluate.
    :type method_list: list[str]
    :return: Two DataFrames containing the R^2 and MAE values for each method.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    r2_list = []
    mae_list = []
    result_df = df.query("dset == 'test'").copy()
    for k, v in result_df.groupby("fold"):
        r2_list.append([r2_score(v[y_col], v[x]) for x in method_list])
        mae_list.append([mean_absolute_error(v[y_col], v[x]) for x in method_list])
    r2_df = pd.DataFrame(r2_list, columns=method_list)
    mae_df = pd.DataFrame(mae_list, columns=method_list)
    r2_melt_df = r2_df.melt()
    r2_melt_df.columns = ["method", "r2"]
    mae_melt_df = mae_df.melt()
    mae_melt_df.columns = ["method", "mae"]
    return r2_melt_df, mae_melt_df

def make_tukey_plot(df, y_col, ax=None, method_col="method", xlim=None, title=""):
    """Draw a Tukey HSD "simultaneous" plot comparing methods on a metric.

    The best-performing method is chosen as the comparison reference: highest
    mean for ``r2`` (higher is better) or lowest mean for ``mae`` (lower is
    better). The x-axis label is set accordingly.

    :param df: Long-format DataFrame with one row per (fold, method) result.
    :param y_col: Metric column to compare. ``"r2"`` and ``"mae"`` are
        recognized and receive appropriate labels and sort directions.
    :param ax: Matplotlib axes to draw on. A new figure/axes is created if
        ``None``.
    :param method_col: Column identifying the method/group for each row.
    :param xlim: Optional ``(xmin, xmax)`` to apply to the x-axis.
    :param title: Title for the axes.
    """
    if ax is None:
        figure, ax = plt.subplots(1, 1)
    tukey = pairwise_tukeyhsd(endog=df[y_col], groups=df[method_col], alpha=0.05)
    ascending = False
    if y_col == "mae":
        ascending = True
    best = df.groupby("method").mean().reset_index().sort_values(y_col, ascending=ascending)[method_col].values[0]
    _ = tukey.plot_simultaneous(comparison_name=best, ax=ax)
    if xlim:
        ax.set_xlim(xlim)
    if y_col == "r2":
        ax.set_xlabel("$R^2$ $\\rightarrow$")
    elif y_col == "mae":
        ax.set_xlabel("$\\leftarrow$ MAE")
    ax.set_title(title)


def plot_r2_mae(r2_combo_df, mae_combo_df, figwidth=15, figheight=5):
    """Plot side-by-side boxplots of R² and MAE across assays and methods.

    :param r2_combo_df: Combined DataFrame with 'assay', 'r2', and 'method' columns.
    :param mae_combo_df: Combined DataFrame with 'assay', 'mae', and 'method' columns.
    :param figwidth: Figure width in inches (default 15).
    :param figheight: Figure height in inches (default 5).
    :return: The matplotlib Figure object.
    """
    figure, axes = plt.subplots(1, 2, figsize=(figwidth, figheight), sharey=True)
    ax = sns.boxplot(y="assay", x="r2", hue="method", data=r2_combo_df, ax=axes[0])
    ax.set_xlabel("$R^2$ $\\rightarrow$")
    ax.set_ylabel("Assay")
    ax.get_legend().remove()
    ax = sns.boxplot(y="assay", x="mae", hue="method", data=mae_combo_df, ax=axes[1])
    ax.set_xlabel("$\\leftarrow$ MAE")
    ax.set_ylabel("Assay")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.close(figure)
    return figure


def plot_tukey_stats(r2_df_list, mae_df_list, assay_list, r2_xlim=(-1, 1), mae_xlim=(0, 1),
                     fig_width=8, fig_height=1, title_height=0.9):
    """Create Tukey HSD comparison plots for multiple assays.

    For each assay a two-panel figure is produced: left panel compares methods by
    R² and right panel compares methods by MAE. The function calls the shared
    Tukey plotting helper for each metric and returns the created figures.

    :param r2_df_list: Sequence of dataframes with 'r2' and 'method' columns.
    :param mae_df_list: Sequence of dataframes with 'mae' and 'method' columns.
    :param assay_list: Sequence of assay names corresponding to the dataframes.
    :param r2_xlim: Optional (xmin, xmax) for R² plots.
    :param mae_xlim: Optional (xmin, xmax) for MAE plots.
    :param fig_width: Width in inches for each figure.
    :param fig_height: Height in inches for each figure.
    :param title_height: Vertical position for the assay title (default 0.9).
    :return: List of matplotlib Figure objects, one per assay.
    """
    figures = []
    for r2_df, mae_df, assay in zip(r2_df_list, mae_df_list, assay_list):
        figure, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)
        axes = axes.flatten()
        uru.make_tukey_plot(r2_df[["r2", "method"]], y_col="r2", ax=axes[0], xlim=r2_xlim)
        uru.make_tukey_plot(mae_df[["mae", "method"]], y_col="mae", ax=axes[1], xlim=mae_xlim)
        figure.suptitle(assay, y=title_height)
        figure.set_size_inches(fig_width, fig_height)
        plt.tight_layout()
        plt.close(figure)
        figures.append(figure)
    return figures



