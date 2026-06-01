import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import useful_rdkit_utils as uru


AMIDATION_SMARTS = "[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:1]NC(=O)[C:2]"


def _named_mol(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    return mol


@pytest.fixture
def reaction():
    return AllChem.ReactionFromSmarts(AMIDATION_SMARTS)


@pytest.fixture
def reagents():
    amines = [_named_mol(smi, f"amine_{i}") for i, smi in enumerate(["CCN", "CCCN", "c1ccccc1N"])]
    acids = [_named_mol(smi, f"acid_{i}") for i, smi in enumerate(["CC(=O)O", "CCC(=O)O"])]
    return [amines, acids]


def test_enumerate_library(reaction, reagents):
    products = uru.enumerate_library(reaction, reagents)
    assert len(products) == 3 * 2
    for smi, name in products:
        assert isinstance(smi, str) and len(smi) > 0
        assert "_" in name


def test_enumerate_library_sample(reaction, reagents):
    sample_df = uru.enumerate_library_sample(reaction, reagents, num_to_generate=4)
    assert isinstance(sample_df, pd.DataFrame)
    assert set(sample_df.columns) == {"SMILES", "Name"}
    assert len(sample_df) >= 4


def test_add_molecule_name():
    smiles = ["CCO", "CCN", "c1ccccc1"]
    names = ["ethanol", "ethylamine", "benzene"]
    df = pd.DataFrame({"SMILES": smiles, "Name": names})
    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    uru.add_molecule_name(df.mol, df.Name)
    for mol, expected_name in zip(df.mol, names):
        assert mol.GetProp("_Name") == expected_name
