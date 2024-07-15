from rdkit import Chem, DataStructs
import useful_rdkit_utils as uru
import numpy as np
from io import StringIO
import pandas as pd


def generate_3D_mol():
    mol_block = """
  Mrv2004 12282122453D

  6  6  0  0  0  0            999 V2000
   -0.0237   -1.2344   -1.9689 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.1104   -1.0717   -0.4320 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1007   -0.3176    0.1775 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3506    1.0411   -0.5283 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4847    0.8784   -2.0653 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.2736    0.1243   -2.6747 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
  4  5  1  0  0  0  0
  5  6  1  0  0  0  0
  1  6  1  0  0  0  0
M  END
        """
    mol = Chem.MolFromMolBlock(mol_block)
    return mol


def test_get_center():
    mol = generate_3D_mol()
    center = uru.get_center(mol)
    ref_center = np.array([-0.68715, -0.09665, -1.24861667])
    dist = np.linalg.norm(center - ref_center)
    assert dist < 0.001


def test_get_shape_moments():
    mol = generate_3D_mol()
    npr1, npr2 = uru.get_shape_moments(mol)
    ref_moments = [0.5231604130266214, 0.52318931029211]
    calc_moments = np.array([npr1, npr2])
    dist = np.linalg.norm(ref_moments - calc_moments)
    assert dist < 0.001


def test_count_fragments():
    mol = Chem.MolFromSmiles("CCC.CC.C")
    num_frags = uru.count_fragments(mol)
    assert num_frags == 3


def test_get_largest_fragment():
    mol = Chem.MolFromSmiles("CCC.CC.C")
    largest_mol = uru.get_largest_fragment(mol)
    largest_smi = Chem.MolToSmiles(largest_mol)
    assert largest_smi == "CCC"


def test_mol2morgan_fp():
    mol_1 = Chem.MolFromSmiles("c1ccccc1")
    fp_1 = uru.mol2morgan_fp(mol_1)
    mol_2 = Chem.MolFromSmiles("c1ccccc1")
    fp_2 = uru.mol2morgan_fp(mol_2)
    assert DataStructs.TanimotoSimilarity(fp_1, fp_2) == 1


def test_mol2numpy_fp():
    mol_1 = Chem.MolFromSmiles("c1ccccc1")
    fp_1 = uru.mol2numpy_fp(mol_1)
    mol_2 = Chem.MolFromSmiles("c1ccccc1")
    fp_2 = uru.mol2numpy_fp(mol_2)
    assert fp_1.sum() == fp_2.sum()


def test_rdkit_props_calc_mol():
    rdkit_props = uru.RDKitProperties()
    mol = Chem.MolFromSmiles("c1ccccc1")
    _ = rdkit_props.calc_mol(mol)
    assert True


def test_rdkit_props_calc_smiles():
    rdkit_props = uru.RDKitProperties()
    _ = rdkit_props.calc_smiles("c1ccccc1")
    assert True


def test_rdkit_descriptors():
    rdkit_desc = uru.RDKitDescriptors()
    smi = "CCCCCC"
    desc_from_smiles = rdkit_desc.calc_smiles(smi)
    mol = Chem.MolFromSmiles(smi)
    desc_from_mol = rdkit_desc.calc_mol(mol)
    assert len(desc_from_smiles == desc_from_mol) == len(rdkit_desc.desc_names)


def test_r05_calc_mol():
    r05_calc = uru.Ro5Calculator()
    mol = Chem.MolFromSmiles("c1ccccc1")
    _ = r05_calc.calc_mol(mol)
    assert True


def test_r05_calc_smiles():
    r05_calc = uru.Ro5Calculator()
    _ = r05_calc.calc_smiles("c1ccccc1")
    assert True


def test_taylor_butina_clustering():
    buff = """C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](OCCOCCOCCOCCOCCOCCOCCOC)CC[C@@]3(O)[C@H]1C5.O=C(O)C(=O)O 1449220
CN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@H]3[C@H]1C5.Cl 699406
COc1ccc2c3c1O[C@H]1[C@@H](O)CC[C@H]4[C@@H](C2)N(C)CC[C@]314.O=C(O)C(O)C(O)C(=O)O 1355857
COc1ccc2c3c1O[C@H]1C(=O)CC[C@H]4[C@@H](C2)N(C)CC[C@]314.O.O.O=C(O)C(O)C(O)C(=O)O 2197548
CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5.CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5.O.O.O.O.O.O=S(=O)(O)O 1376014
Cn1c(=O)c2c(ncn2C)n(C)c1=O 16485
Cn1c(=O)c2nc[nH]c2n(C)c1=O 381693
Cn1c(=O)c2[nH]cnc2n(C)c1=O.Cn1c(=O)c2[nH]cnc2n(C)c1=O.NCCN 794445
Cn1c(=O)c2[nH]cnc2n(C)c1=O.NCC(=O)[O-].[Na+] 674529
C[N+](C)(C)CCO.Cn1c(=O)c2[n-]cnc2n(C)c1=O 674385"""
    ifs = StringIO(buff)
    df = pd.read_csv(ifs, sep=" ", names=["SMILES", "Name"])
    df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
    df["fp"] = df.mol.apply(uru.mol2morgan_fp)
    clusters = uru.taylor_butina_clustering(df.fp.values)
    assert str(clusters) == '[3, 1, 1, 1, 1, 0, 0, 0, 0, 2]'


def test_label_atoms():
    mol = Chem.MolFromSmiles("c1ccccc1")
    uru.label_atoms(mol, range(0, mol.GetNumAtoms()))
    atom_labels = [atm.GetProp("atomNote") for atm in mol.GetAtoms()]
    assert str(atom_labels) == "['0', '1', '2', '3', '4', '5']"
    assert True


def test_tag_atoms():
    mol = Chem.MolFromSmiles("c1ccccc1")
    idx_list = [0, 1, 2]
    uru.tag_atoms(mol, idx_list, "X")
    atm_list = [mol.GetAtomWithIdx(i) for i in idx_list]
    tag_list = [atm.GetProp("atomNote") for atm in atm_list]
    assert str(tag_list) == "['X', 'X', 'X']"


def test_gen_3d():
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = uru.gen_3d(mol)
    assert mol.GetNumConformers() == 1


def test_get_spiro_atoms():
    mol = Chem.MolFromSmiles("C1CC2(C1)CCC1(CCCC1)CC2")
    spiro_atoms = uru.get_spiro_atoms(mol)
    assert str(spiro_atoms) == "[2, 6]"


def test_max_ring_size():
    mol = Chem.MolFromSmiles("C1CC2(C1)CCC1(CCCC1)CC2")
    max_ring_size = uru.max_ring_size(mol)
    assert max_ring_size == 6
