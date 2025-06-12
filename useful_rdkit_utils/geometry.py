from typing import Optional, List

import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors3D import NPR1, NPR2
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.Chem.rdchem import Mol
from typing import Tuple
from itertools import combinations


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


def gen_conformers(mol: Mol, num_confs: int = 50) -> Optional[Mol]:
    """
    Generate conformers for a molecule.

    :param mol: RDKit molecule
    :param num_confs: Number of conformers to generate
    :return: Molecule with conformers or None if generation fails
    """
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        confgen_res = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        if len(confgen_res) != num_confs:
            raise ValueError(f"Failed to generate {num_confs} conformers, got {len(confgen_res)}")
        energy_list = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
        for energy_tuple, conf in zip(energy_list, mol.GetConformers()):
            _, energy = energy_tuple
            conf.SetDoubleProp("Energy", energy)
        mol = Chem.RemoveHs(mol)
        return mol
    except Exception as e:
        print(f"Error generating conformers: {e}")
        return None

def refine_conformers(mol: Mol, energy_threshold: float = 50, rms_threshold: Optional[float] = 0.5) -> Mol:
    """
    Refine the conformers of a molecule by removing those with high energy or low RMSD.

    :param mol: RDKit molecule with conformers.
    :param energy_threshold: Energy threshold above which conformers are removed.
    :param rms_threshold: RMSD threshold below which conformers are considered redundant and removed.
                          If None, RMSD filtering is skipped.
    :return: RDKit molecule with refined conformers.
    """
    energy_list = [None] * mol.GetNumConformers()
    for i in range(0, mol.GetNumConformers()):
        conf = mol.GetConformer(i)
        conf_idx = conf.GetId()
        energy_list[conf_idx] = float(conf.GetProp("Energy"))
    energy_array = np.array(energy_list)
    min_energy = min(energy_list)
    energy_array -= min_energy
    energy_remove_idx = np.argwhere(energy_array > energy_threshold)
    for i in energy_remove_idx.flatten()[::-1]:
        mol.RemoveConformer(int(i))
    conf_ids = [x.GetId() for x in mol.GetConformers()]

    if rms_threshold is not None:
        rms_list = [(i1, i2, AllChem.GetConformerRMS(mol, i1, i2)) for i1, i2 in combinations(conf_ids, 2)]
        rms_remove_idx = list(set([x[1] for x in rms_list if x[2] < rms_threshold]))
        rms_remove_idx.reverse()
        for i in rms_remove_idx:
            mol.RemoveConformer(int(i))
    return mol

def get_conformer_energies(mol: Mol) -> List[float]:
    """
    Retrieve the energies of all conformers in a molecule.

    :param mol: RDKit molecule containing conformers.
    :return: A list of energies for each conformer as floats.
    """
    return [float(conf.GetProp("Energy")) for conf in mol.GetConformers()]


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


# Adapted from https://birdlet.github.io/2019/10/02/py3dmol_example/
def mol_to_3D_view(mol_list, size=(300, 300), style="stick", surface=False, opacity=0.5) -> py3Dmol.view:
    """Draw a list of molecules in 3D

    :mol_list: list[rdMol], a list of rdMols to show
    :size: tuple(int, int), canvas size
    :style: str, type of drawing molecule,
        style can be 'line', 'stick', 'sphere', 'carton'
    :surface: bool, display SAS
    :opacity: float, opacity of surface, range 0.0-1.0
    :return: viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    import py3Dmol
    assert style in ('line', 'stick', 'sphere', 'cartoon')

    colors = ["lightgray", "pink", "lightgreen", "magenta", "cyan", "orange", "purple"]

    viewer = py3Dmol.view(width=size[0], height=size[1])

    for i, mol in enumerate(mol_list):
        color_idx = i % len(colors)
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
        viewer.setStyle({'model': i}, {'stick': {'colorscheme': f'{colors[color_idx]}Carbon'}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
