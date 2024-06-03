from typing import Optional

import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors3D import NPR1, NPR2
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdchem import Mol


# ----------- Molecular geometry
def get_center(mol) -> np.ndarray:
    """Get the geometric center of an RDKit molecule

    :param mol: RDKit molecule
    :return: center as a numpy array
    """
    assert mol.GetNumConformers() > 0, "Molecule must have at least one conformer"
    return np.array(ComputeCentroid(mol.GetConformer(0)))


def get_shape_moments(mol) -> tuple:
    """ Calculate principal moments of inertia as defined in https://pubs.acs.org/doi/10.1021/ci025599w

    :param mol: RDKit molecule
    :return: first 2 moments as a tuple
    """
    assert mol.GetNumConformers() > 0, "molecule must have at least one conformer"
    npr1 = NPR1(mol)
    npr2 = NPR2(mol)
    return npr1, npr2


# ----------- 3D related stuff
def gen_3d(mol: Mol) -> Optional[Mol]:
    """Generate a 3D structure for a RDKit molecule

    :param mol: input molecule
    :return: molecule with 3D coordinates
    """
    mol_3d = gen_conformers(mol, num_confs=1)
    return mol_3d


def gen_conformers(mol, num_confs=10):
    """Generate conformers for a molecule

    :param mol: RDKit molecule
    :return: molecule with conformers
    """
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        mol = Chem.RemoveHs(mol)
    except ValueError:
        mol = None
    return mol


def mcs_rmsd(mol_1: Mol, mol_2: Mol) -> float:
    """
    Calculate the minimum Root Mean Square Deviation (RMSD) between the Maximum Common Substructure (MCS)
    of two RDKit molecule objects.

    :param mol_1: The first RDKit molecule object.
    :param mol_2: The second RDKit molecule object.
    :return: The minimum RMSD as a float.
    """
    mcs_res = FindMCS([mol_1, mol_2])
    pat = Chem.MolFromSmarts(mcs_res.smartsString)
    match_1 = mol_1.GetSubstructMatches(pat)
    match_2 = mol_2.GetSubstructMatches(pat)
    min_rmsd = 1e6
    for m1 in match_1:
        for m2 in match_2:
            crd_1 = mol_1.GetConformer(0).GetPositions()[list(m1)]
            crd_2 = mol_2.GetConformer(0).GetPositions()[list(m2)]
            rmsd = np.sqrt((((crd_1 - crd_2) ** 2).sum()).mean())
            min_rmsd = min(min_rmsd, rmsd)
    return min_rmsd


# from https://birdlet.github.io/2019/10/02/py3dmol_example/
def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5) -> py3Dmol.view:
    """Draw molecule in 3D

    :mol: rdMol, molecule to show
    :size: tuple(int, int), canvas size
    :style: str, type of drawing molecule,
        style can be 'line', 'stick', 'sphere', 'carton'
    :surface: bool, display SAS
    :opacity: float, opacity of surface, range 0.0-1.0
    :return: viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    import py3Dmol
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
