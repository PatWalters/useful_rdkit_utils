import pandas as pd
import pytest
from rdkit import Chem

import useful_rdkit_utils as uru

# REOS downloads its alert collection from GitHub when it is constructed, so every
# test here needs the network. Run the rest of the suite with -m 'not network'.
pytestmark = pytest.mark.network


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


def test_set_min_priority():
    """set_min_priority() previously raised because it referenced an undefined name."""
    reos = uru.REOS(active_rules=["Glaxo", "Dundee", "Inpharmatica"])
    num_rules_before = len(reos.active_rule_df)

    # Glaxo has priority 8, Dundee 4, Inpharmatica 1
    reos.set_min_priority(4)
    assert set(reos.get_active_rule_sets()) == {"Glaxo", "Dundee"}
    assert len(reos.active_rule_df) < num_rules_before

    # a second call resets from the selected rule sets rather than narrowing further
    reos.set_min_priority(1)
    assert set(reos.get_active_rule_sets()) == {"Glaxo", "Dundee", "Inpharmatica"}
    assert len(reos.active_rule_df) == num_rules_before


def test_set_min_priority_after_set_active_rule_sets():
    """The priority filter follows the most recent rule set selection."""
    reos = uru.REOS()
    reos.set_active_rule_sets(["Dundee"])
    reos.set_min_priority(1)
    assert set(reos.get_active_rule_sets()) == {"Dundee"}
    reos.set_min_priority(8)
    assert len(reos.active_rule_df) == 0


def test_set_active_rule_sets_accepts_a_bare_string():
    """A single rule set name is a natural argument; it must not be split into characters."""
    reos = uru.REOS()
    reos.set_active_rule_sets("PAINS")
    assert list(reos.get_active_rule_sets()) == ["PAINS"]
    assert reos.active_rules == ["PAINS"]
    # the stored value has to survive into a later priority filter
    reos.set_min_priority(1)
    assert list(reos.get_active_rule_sets()) == ["PAINS"]


def test_read_rules_accepts_a_bare_string():
    reos = uru.REOS()
    reos.read_rules(reos.get_rule_file_location(), "Dundee")
    assert list(reos.get_active_rule_sets()) == ["Dundee"]
