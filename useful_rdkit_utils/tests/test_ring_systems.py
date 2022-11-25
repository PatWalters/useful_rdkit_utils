import useful_rdkit_utils as uru
from rdkit import Chem


def test_ring_system_finder():
    mol = Chem.MolFromSmiles("CC(=O)[O-].CCn1c(=O)/c(=C2\Sc3ccccc3N2C)s/c1=C\C1CCC[n+]2c1sc1ccccc12")
    ring_system_finder = uru.RingSystemFinder()
    res = ring_system_finder.find_ring_systems(mol)
    assert res[0] == 'C=c1[nH]c(=O)/c(=C2/Nc3ccccc3S2)s1' and res[1] == 'c1ccc2c(c1)sc1[n+]2CCCC1'

