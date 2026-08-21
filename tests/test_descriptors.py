import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import useful_rdkit_utils as uru


@pytest.fixture
def property_df():
    """A properties table of the shape plot_properties() is meant to receive."""
    return pd.DataFrame({
        "MolWt": [180.2, 194.2, 122.1, 138.1],
        "LogP": [1.2, 1.8, 1.9, 1.5],
        "TPSA": [63.6, 63.6, 37.3, 57.5],
    })


def test_plot_properties_multiple_columns(property_df):
    """One histogram per column; indexing the (1, n) axes array must use both indices."""
    figure = uru.plot_properties(property_df)
    try:
        assert len(figure.axes) == len(property_df.columns)
        assert [ax.get_title() for ax in figure.axes] == list(property_df.columns)
    finally:
        plt.close(figure)


def test_plot_properties_single_column(property_df):
    figure = uru.plot_properties(property_df[["MolWt"]])
    try:
        assert len(figure.axes) == 1
        assert figure.axes[0].get_title() == "MolWt"
    finally:
        plt.close(figure)


def test_clean_descriptors_dataframe():
    """A DataFrame in gives a DataFrame out, with the surviving column labels."""
    df = pd.DataFrame({
        "keep_1": [1.0, 2.0, 3.0],
        "has_nan": [1.0, np.nan, 3.0],
        "keep_2": [4.0, 9.0, 2.0],
        "has_inf": [1.0, np.inf, 3.0],
    })
    clean, kept_indices = uru.clean_descriptors(df)
    assert isinstance(clean, pd.DataFrame)
    assert list(clean.columns) == ["keep_1", "keep_2"]
    assert kept_indices == [0, 2]


def test_clean_descriptors_ndarray():
    """An ndarray in still gives an ndarray out."""
    arr = np.array([[1.0, 1.0, 4.0],
                    [2.0, np.nan, 9.0],
                    [3.0, 3.0, 2.0]])
    clean, kept_indices = uru.clean_descriptors(arr)
    assert isinstance(clean, np.ndarray)
    assert clean.shape == (3, 2)
    assert kept_indices == [0, 2]


def test_clean_descriptors_all_columns_bad():
    with pytest.raises(ValueError):
        uru.clean_descriptors(np.array([[np.nan, 1.0], [1.0, np.inf]]))


def test_clean_and_scale_descriptors_dataframe():
    """The annotated input type -- a DataFrame -- must be accepted."""
    df = pd.DataFrame({
        "keep_1": [1.0, 2.0, 3.0, 4.0],
        "has_nan": [1.0, np.nan, 3.0, 4.0],
        "keep_2": [4.0, 9.0, 2.0, 7.0],
    })
    scaled, scaler = uru.clean_and_scale_descriptors(df)
    assert scaled.shape == (4, 2)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)
    # the scaler is fitted and reusable on new data with the cleaned column count
    assert scaler.transform(df[["keep_1", "keep_2"]]).shape == (4, 2)


def test_clean_and_scale_descriptors_ndarray():
    arr = np.array([[1.0, 1.0, 4.0],
                    [2.0, np.nan, 9.0],
                    [3.0, 3.0, 2.0],
                    [4.0, 4.0, 7.0]])
    scaled, _ = uru.clean_and_scale_descriptors(arr)
    assert scaled.shape == (4, 2)
