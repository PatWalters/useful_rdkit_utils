import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

import useful_rdkit_utils as uru


SMILES_LIST = [
    "c1ccccc1", "c1ccccc1C", "c1ccccc1CC", "CCCCCCCC", "O=C(O)c1ccccc1",
    "CCN", "CCO", "CCCl", "c1ccccc1O", "c1ccccc1N",
]


@pytest.fixture
def descriptor_dict():
    rng = np.random.default_rng(0)
    return {smi: rng.random(8) for smi in SMILES_LIST}


@pytest.fixture
def split_frames():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({
        "SMILES": SMILES_LIST,
        "y": rng.random(len(SMILES_LIST)),
    })
    test = pd.DataFrame({
        "SMILES": SMILES_LIST[:5],
        "y": rng.random(5),
    })
    return train, test


def test_wrapper_factory_creates_class(descriptor_dict):
    Wrapper = uru.WrapperFactory.create_wrapper_class(
        LinearRegression, descriptor_dict, class_name="LRWrapper"
    )
    assert Wrapper.__name__ == "LRWrapper"
    wrapper = Wrapper(y_col="y")
    assert isinstance(wrapper.model, LinearRegression)
    assert wrapper.y_col == "y"


def test_wrapper_factory_fit_predict(descriptor_dict, split_frames):
    train, test = split_frames
    Wrapper = uru.WrapperFactory.create_wrapper_class(LinearRegression, descriptor_dict)
    wrapper = Wrapper(y_col="y")
    wrapper.fit(train)
    pred = wrapper.predict(test)
    assert len(pred) == len(test)


def test_wrapper_factory_validate(descriptor_dict, split_frames):
    train, test = split_frames
    Wrapper = uru.WrapperFactory.create_wrapper_class(LinearRegression, descriptor_dict)
    wrapper = Wrapper(y_col="y")
    pred = wrapper.validate(train, test)
    assert len(pred) == len(test)


def test_wrapper_factory_forwards_kwargs(descriptor_dict):
    Wrapper = uru.WrapperFactory.create_wrapper_class(
        Ridge, descriptor_dict, alpha=0.5
    )
    wrapper = Wrapper(y_col="y")
    assert wrapper.model.alpha == 0.5


@pytest.fixture
def cv_results():
    rng = np.random.default_rng(2)
    rows = []
    for fold in range(4):
        for dset in ["train", "test"]:
            for _ in range(10):
                rows.append({
                    "fold": fold,
                    "dset": dset,
                    "y": rng.random(),
                    "method_a": rng.random(),
                    "method_b": rng.random(),
                    "method_c": rng.random(),
                })
    return pd.DataFrame(rows)


def test_get_performance_stats(cv_results):
    r2_df, mae_df = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    assert set(r2_df.columns) == {"method", "r2"}
    assert set(mae_df.columns) == {"method", "mae"}
    # 4 folds * 3 methods
    assert len(r2_df) == 12
    assert len(mae_df) == 12
    assert (mae_df["mae"] >= 0).all()


def test_make_tukey_plot_r2(cv_results):
    r2_df, _ = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    fig, ax = plt.subplots()
    uru.make_tukey_plot(r2_df, y_col="r2", ax=ax, title="R2 comparison")
    assert ax.get_xlabel() == "$R^2$ $\\rightarrow$"
    assert ax.get_title() == "R2 comparison"
    plt.close(fig)


def test_make_tukey_plot_mae_with_xlim(cv_results):
    _, mae_df = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    fig, ax = plt.subplots()
    uru.make_tukey_plot(mae_df, y_col="mae", ax=ax, xlim=(0, 2))
    assert ax.get_xlabel() == "$\\leftarrow$ MAE"
    assert ax.get_xlim() == (0.0, 2.0)
    plt.close(fig)


def test_make_tukey_plot_creates_axes_if_none(cv_results):
    r2_df, _ = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    # Should not raise when ax is None
    uru.make_tukey_plot(r2_df, y_col="r2")
    plt.close("all")


def test_make_tukey_plot_custom_method_col(cv_results):
    r2_df, _ = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    df = r2_df.rename(columns={"method": "approach"})
    fig, ax = plt.subplots()
    uru.make_tukey_plot(df, y_col="r2", ax=ax, method_col="approach")
    plt.close(fig)


def test_plot_r2_mae(cv_results):
    r2_df, mae_df = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b"]
    )
    r2_df = r2_df.assign(assay="A1")
    mae_df = mae_df.assign(assay="A1")
    fig = uru.plot_r2_mae(r2_df, mae_df, figwidth=8, figheight=4)
    assert fig is not None
    assert fig.get_size_inches()[0] == pytest.approx(8)


def test_plot_tukey_stats_returns_figures(cv_results):
    r2_a, mae_a = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    r2_b, mae_b = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    figures = uru.plot_tukey_stats(
        [r2_a, r2_b], [mae_a, mae_b], assay_list=["A1", "A2"]
    )
    assert len(figures) == 2
    assert all(fig is not None for fig in figures)


def test_plot_tukey_stats_custom_xlim(cv_results):
    r2_a, mae_a = uru.get_performance_stats(
        cv_results, y_col="y", method_list=["method_a", "method_b", "method_c"]
    )
    figures = uru.plot_tukey_stats(
        [r2_a], [mae_a], assay_list=["A1"], r2_xlim=(-1, 1), mae_xlim=(0, 1.5)
    )
    assert len(figures) == 1
    assert figures[0] is not None
