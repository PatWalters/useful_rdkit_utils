import pandas as pd
import pytest
from rdkit import Chem

import useful_rdkit_utils as uru


@pytest.fixture
def small_series():
    """A small set of biaryl-like molecules sharing a common scaffold."""
    smiles_names = [
        ("c1ccc(-c2ccccc2)cc1", "mol_a"),
        ("Cc1ccc(-c2ccccc2)cc1", "mol_b"),
        ("Clc1ccc(-c2ccccc2)cc1", "mol_c"),
        ("Oc1ccc(-c2ccccc2)cc1", "mol_d"),
    ]
    df = pd.DataFrame(smiles_names, columns=["SMILES", "Name"])
    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    df["pIC50"] = [5.0, 6.1, 7.2, 4.8]
    return df


def test_cleanup_fragment():
    mol = Chem.MolFromSmiles("[*:1]c1ccccc1[*:2]")
    cleaned, rgroup_count = uru.cleanup_fragment(mol)
    assert rgroup_count == 2
    assert Chem.MolToSmiles(cleaned) == "c1ccccc1"


def test_generate_fragments():
    mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
    frag_df = uru.generate_fragments(mol)
    assert isinstance(frag_df, pd.DataFrame)
    assert set(frag_df.columns) == {"Scaffold", "NumAtoms", "NumRgroups"}
    # The whole molecule should be in the fragment list
    assert Chem.MolToSmiles(mol) in frag_df["Scaffold"].tolist()


def test_find_scaffolds(small_series):
    mol_df, scaffold_df = uru.find_scaffolds(small_series, disable_progress=True)
    assert isinstance(mol_df, pd.DataFrame)
    assert isinstance(scaffold_df, pd.DataFrame)
    assert set(scaffold_df.columns) == {"Scaffold", "Count", "NumAtoms"}
    # Biphenyl should be the most frequent scaffold
    top_scaffold = scaffold_df.iloc[0]["Scaffold"]
    assert Chem.MolFromSmiles(top_scaffold) is not None
    assert scaffold_df.iloc[0]["Count"] >= 2


def test_get_molecules_with_scaffold(small_series):
    mol_df, scaffold_df = uru.find_scaffolds(small_series, disable_progress=True)
    top_scaffold = scaffold_df.iloc[0]["Scaffold"]
    cores, match_df = uru.get_molecules_with_scaffold(
        top_scaffold, mol_df, small_series
    )
    assert isinstance(match_df, pd.DataFrame)
    assert "pIC50" in match_df.columns
    assert len(match_df) >= 1


def test_get_molecules_with_scaffold_no_match(small_series):
    mol_df, _ = uru.find_scaffolds(small_series, disable_progress=True)
    cores, match_df = uru.get_molecules_with_scaffold(
        "c1ccncc1",  # pyridine — not present
        mol_df,
        small_series,
    )
    assert list(cores) == []
    assert len(match_df) == 0
