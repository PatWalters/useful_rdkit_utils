from typing import Optional

import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors3D import NPR1, NPR2
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdchem import Mol
from typing import Tuple


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


def mcs_rmsd(mol_1: Mol, mol_2: Mol, id_1: int = 0, id_2: int = 0) -> Tuple[int, float]:
    """
    Calculate the RMSD (Root Mean Square Deviation) between the MCS (Maximum Common Substructure) of two molecules.

    :param mol_1: First RDKit molecule
    :param mol_2: Second RDKit molecule
    :param id_1: Conformer ID for the first molecule
    :param id_2: Conformer ID for the second molecule
    :return: A tuple containing the number of MCS atoms and the RMSD value
    """
    mcs_res = FindMCS([mol_1, mol_2])
    num_mcs_atoms = mcs_res.numAtoms
    pat = Chem.MolFromSmarts(mcs_res.smartsString)
    match_1 = mol_1.GetSubstructMatches(pat)
    match_2 = mol_2.GetSubstructMatches(pat)
    min_rmsd = 1e6
    for m1 in match_1:
        for m2 in match_2:
            crd_1 = mol_1.GetConformer(id_1).GetPositions()[list(m1)]
            crd_2 = mol_2.GetConformer(id_2).GetPositions()[list(m2)]
            diff = crd_1 - crd_2
            squared_dist = np.sum(diff ** 2, axis=1)
            msd = np.mean(squared_dist)
            rmsd = np.sqrt(msd)
            min_rmsd = min(min_rmsd, rmsd)
    return num_mcs_atoms, float(min_rmsd)


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
