import numpy as np
import pandas as pd
import pytest

import useful_rdkit_utils as uru
from useful_rdkit_utils.split_utils import (
    GroupKFoldShuffle,
    audit_split_leakage,
    summarize_split_leakage,
)


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


# A training set and a test set with leakage planted deliberately, one kind per row.
LEAKY_TRAIN = [
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",    # ibuprofen
    "C[C@H](N)C(=O)O",               # L-alanine
    "CCN",                           # ethylamine, as the free base
    "c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
]


def test_audit_split_leakage_finds_an_exact_duplicate():
    audit = audit_split_leakage(LEAKY_TRAIN, ["CC(=O)Oc1ccccc1C(=O)O"])
    assert bool(audit.loc[0, "exact"])
    assert audit.loc[0, "nearest_similarity"] == pytest.approx(1.0)
    assert audit.loc[0, "nearest_train_smiles"] == "CC(=O)Oc1ccccc1C(=O)O"


def test_audit_split_leakage_finds_an_enantiomer():
    """An enantiomer of a training compound is a different molecule but still leakage.

    The InChIKey differs, so the exact check misses it; the first block is
    constitution only, so the connectivity check catches it.
    """
    audit = audit_split_leakage(LEAKY_TRAIN, ["C[C@@H](N)C(=O)O"])
    assert not bool(audit.loc[0, "exact"])
    assert bool(audit.loc[0, "same_connectivity"])


def test_audit_split_leakage_sees_through_a_salt():
    """A salt of a training compound counts, and only once the counter ion is gone."""
    stripped = audit_split_leakage(LEAKY_TRAIN, ["CCN.Cl"], strip_salts=True)
    intact = audit_split_leakage(LEAKY_TRAIN, ["CCN.Cl"], strip_salts=False)
    assert bool(stripped.loc[0, "same_connectivity"])
    assert not bool(intact.loc[0, "same_connectivity"])


def test_audit_split_leakage_leaves_a_genuinely_new_molecule_alone():
    audit = audit_split_leakage(LEAKY_TRAIN, ["c1ccc2ccccc2c1", "O=C(O)CCCCCCCCCCCCCCCCC"])
    assert not audit["exact"].any()
    assert not audit["same_connectivity"].any()
    assert (audit["nearest_similarity"] < 0.5).all()


def test_audit_split_leakage_flags_a_near_duplicate_at_the_cutoff():
    """The methyl ester of a training acid is not the same molecule, but it is close."""
    ester = "CC(C)Cc1ccc(cc1)C(C)C(=O)OC"
    audit = audit_split_leakage(LEAKY_TRAIN, [ester], similarity_cutoff=0.6)
    assert not bool(audit.loc[0, "same_connectivity"])
    assert bool(audit.loc[0, "near_duplicate"])
    strict = audit_split_leakage(LEAKY_TRAIN, [ester], similarity_cutoff=0.95)
    assert not bool(strict.loc[0, "near_duplicate"])


def test_audit_split_leakage_survives_an_unparsable_smiles():
    """One bad row must not take the audit down with it."""
    audit = audit_split_leakage(LEAKY_TRAIN, ["not a molecule", "c1ccccc1"])
    assert not bool(audit.loc[0, "exact"])
    assert not bool(audit.loc[0, "same_connectivity"])
    assert np.isnan(audit.loc[0, "nearest_similarity"])
    assert audit.loc[0, "nearest_train_smiles"] is None
    assert bool(audit.loc[1, "exact"])


def test_audit_split_leakage_tolerates_a_bad_training_smiles():
    audit = audit_split_leakage(["not a molecule"] + LEAKY_TRAIN, ["c1ccccc1"])
    assert bool(audit.loc[0, "exact"])


def test_audit_split_leakage_returns_one_row_per_test_molecule():
    test_smiles = ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    audit = audit_split_leakage(LEAKY_TRAIN, test_smiles)
    assert len(audit) == len(test_smiles)
    assert list(audit["test_smiles"]) == test_smiles


def test_summarize_split_leakage_reports_fractions():
    test_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "C[C@@H](N)C(=O)O", "c1ccc2ccccc2c1", "CCO"]
    summary = summarize_split_leakage(audit_split_leakage(LEAKY_TRAIN, test_smiles))
    assert summary["n_test"] == 4
    assert summary["exact"] == pytest.approx(0.25)
    assert summary["same_connectivity"] == pytest.approx(0.5)
    assert 0.0 <= summary["median_nearest_similarity"] <= 1.0
