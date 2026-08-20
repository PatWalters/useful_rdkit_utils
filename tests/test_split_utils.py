import numpy as np
import pandas as pd
import pytest

import useful_rdkit_utils as uru
from useful_rdkit_utils.split_utils import GroupKFoldShuffle


class _FakeModel:
    """Minimal stand-in for the WrapperFactory-created models."""

    def __init__(self, y_col):
        self.y_col = y_col

    def validate(self, train, test):
        return np.full(len(test), train[self.y_col].mean())


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    smiles = ["c1ccccc1", "CCO", "CCN", "CCOC", "c1ccc(O)cc1", "c1ccc(N)cc1",
              "Cc1ccccc1", "NCc1ccccc1", "OCC1=CC=CC=C1", "CC(C)O"] * 4
    n = len(smiles)
    return pd.DataFrame({"SMILES": smiles, "row_id": np.arange(n), "y": rng.normal(size=n)})


def test_group_kfold_shuffle_error_when_too_many_splits():
    kf = GroupKFoldShuffle(n_splits=10, shuffle=True)
    with pytest.raises(ValueError):
        _ = list(kf.split(X=np.zeros(5), groups=[0, 1, 2]))


def test_cross_validate_nested(small_df):
    n_outer, n_inner = 3, 2
    model_list = [("fake", _FakeModel)]
    group_list = [("random", uru.get_random_clusters)]
    res = uru.cross_validate(small_df, model_list, "y", group_list,
                             n_outer=n_outer, n_inner=n_inner, random_state=1)

    assert "fake" in res.columns
    assert {"dset", "group", "fold"} <= set(res.columns)

    test_df = res.query("dset == 'test'")
    folds_seen = set(test_df["fold"].unique())
    # per outer round: n_inner inner folds plus the outer test evaluation
    expected = {i * (n_inner + 1) + k for i in range(n_outer) for k in range(n_inner + 1)}
    assert folds_seen == expected

    # no leakage: the outer test rows must not appear in the inner training rows of the same round
    for i in range(n_outer):
        inner_folds = set(range(i * (n_inner + 1), i * (n_inner + 1) + n_inner))
        train_rows = res.query("dset == 'train'")
        train_rows = train_rows[train_rows["fold"].isin(inner_folds)]
        outer_test_folds = {i * (n_inner + 1) + n_inner}
        outer_rows = test_df[test_df["fold"].isin(outer_test_folds)]
        assert not set(train_rows["row_id"]) & set(outer_rows["row_id"])

    # every test row has a model prediction, and each fold label has at least one test row
    assert test_df["fake"].notna().all()
    assert test_df.groupby("fold").ngroups == n_outer * (n_inner + 1)


def test_get_scaffold_acyclic():
    assert uru.get_scaffold("CCCC") == "CCCC"
