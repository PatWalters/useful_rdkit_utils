import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from useful_rdkit_utils.geometry import (
    get_center,
    get_shape_moments,
    gen_3d,
    gen_conformers,
    refine_conformers,
    get_conformer_energies,
    mcs_rmsd,
    mol_to_3D_view,
)
import py3Dmol

@pytest.fixture
def benzene():
    """Fixture for a benzene molecule."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

@pytest.fixture
def ethanol():
    """Fixture for an ethanol molecule."""
    mol = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    return mol

def test_get_center(benzene):
    center = get_center(benzene)
    assert isinstance(center, np.ndarray)
    assert center.shape == (3,)

def test_get_shape_moments(benzene):
    npr1, npr2 = get_shape_moments(benzene)
    assert isinstance(npr1, float)
    assert isinstance(npr2, float)
    assert 0 <= npr1 <= 1
    assert 0 <= npr2 <= 1

def test_gen_3d(ethanol):
    mol_3d = gen_3d(ethanol)
    assert mol_3d is not None
    assert mol_3d.GetNumConformers() > 0

def test_gen_conformers(ethanol):
    mol_with_confs = gen_conformers(ethanol, num_confs=5)
    assert mol_with_confs is not None
    assert mol_with_confs.GetNumConformers() == 5

def test_refine_conformers(ethanol):
    mol_with_confs = gen_conformers(ethanol, num_confs=10)
    refined_mol = refine_conformers(mol_with_confs, energy_threshold=10, rms_threshold=0.5)
    assert refined_mol.GetNumConformers() <= 10

def test_get_conformer_energies(ethanol):
    mol_with_confs = gen_conformers(ethanol, num_confs=5)
    energies = get_conformer_energies(mol_with_confs)
    assert len(energies) == 5
    assert all(isinstance(e, float) for e in energies)

def test_mcs_rmsd(benzene, ethanol):
    num_mcs_atoms, rmsd = mcs_rmsd(benzene, ethanol)
    assert isinstance(num_mcs_atoms, int)
    assert isinstance(rmsd, float)
    assert num_mcs_atoms >= 0
    assert rmsd >= 0

def test_mol_to_3D_view(benzene, ethanol):
    viewer = mol_to_3D_view([benzene, ethanol], size=(400, 400), style="stick", surface=True, opacity=0.7)
    assert viewer is not None
    assert isinstance(viewer, py3Dmol.view)