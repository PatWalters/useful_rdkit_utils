import pandas as pd
from rdkit import Chem

import useful_rdkit_utils as uru


def test_parse_smarts():
    reos = uru.REOS()
    assert reos.parse_smarts()


def test_process_mol():
    smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    mol = Chem.MolFromSmiles(smiles)
    reos = uru.REOS()
    res = reos.process_mol(mol)
    assert str(res) == "('ok', 'ok')"


def test_process_smiles():
    smiles = "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C"
    reos = uru.REOS()
    res = reos.process_smiles(smiles)
    assert str(res) == "('ok', 'ok')"


def test_pandas_smiles():
    reos = uru.REOS()
    reos.output_smarts = True
    smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=CC=C(C=C1)C(=O)O", "C1=CC=C(C=C1)C(=O)O"]
    result = reos.pandas_smiles(smiles_list)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(smiles_list)
    assert set(result.columns) == {'rule_set_name', 'description', 'smarts'}


def test_pandas_mols():
    reos = uru.REOS()
    reos.output_smarts = True
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in
                ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC1=CC=C(C=C1)C(=O)O", "C1=CC=C(C=C1)C(=O)O"]]
    result = reos.pandas_mols(mol_list)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(mol_list)
    assert set(result.columns) == {'rule_set_name', 'description', 'smarts'}
