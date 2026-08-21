import pandas as pd
import pytest

import useful_rdkit_utils as uru


TRAINING_SMILES = [
    "c1ccccc1CCN",
    "CCOc1ccccc1",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
]


@pytest.fixture
def silly_walks():
    sw = uru.SillyWalks()
    sw.build_dict(pd.DataFrame({"canonical_smiles": TRAINING_SMILES}))
    return sw


def test_silly_walks_is_exported():
    """SillyWalks must be reachable from the package namespace, like every other class."""
    assert hasattr(uru, "SillyWalks")
    assert uru.SillyWalks.__module__ == "useful_rdkit_utils.silly_walks"


def test_silly_walks_scores_training_molecule_as_zero(silly_walks):
    """Every bit of a training molecule is known, so nothing is silly."""
    assert silly_walks.score(TRAINING_SMILES[0]) == 0.0


def test_silly_walks_scores_unseen_chemistry_above_zero(silly_walks):
    """A molecule built from unseen environments has bits missing from the dictionary."""
    assert silly_walks.score("[SiH3][Ge](F)(F)[Se]C1=NN=NN1") > 0.0


def test_silly_walks_scores_unparsable_smiles_as_one(silly_walks):
    assert silly_walks.score("not_a_smiles") == 1


def test_silly_walks_build_dict_accumulates(silly_walks):
    assert len(silly_walks.count_dict) > 0
    assert all(isinstance(k, int) and v > 0 for k, v in silly_walks.count_dict.items())
