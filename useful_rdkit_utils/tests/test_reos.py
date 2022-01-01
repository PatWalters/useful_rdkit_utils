import useful_rdkit_utils as uru
from rdkit import Chem


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
