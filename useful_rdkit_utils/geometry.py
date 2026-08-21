import warnings
from typing import Optional, List

import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
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


def gen_conformers(mol: Mol, num_confs: int = 50) -> Mol:
    """
    Generate conformers for a molecule.

    A molecule that is too rigid to support ``num_confs`` distinct conformers yields
    however many could be embedded, with a warning; that is a property of the molecule
    rather than an error. Genuine failures raise, so that they can be told apart from
    a small conformer count.

    :param mol: RDKit molecule
    :param num_confs: Number of conformers to generate
    :raises ValueError: if no conformers could be embedded, or if MMFF parameters are
        not available for the molecule so conformer energies cannot be assigned
    :return: Molecule with conformers, each carrying an "Energy" property
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    confgen_res = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    if len(confgen_res) == 0:
        raise ValueError(f"Could not embed any conformers for {Chem.MolToSmiles(Chem.RemoveHs(mol))}")
    if len(confgen_res) != num_confs:
        warnings.warn(
            f"Requested {num_confs} conformers, embedded {len(confgen_res)}", stacklevel=2
        )
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError(
            f"MMFF parameters are not available for {Chem.MolToSmiles(Chem.RemoveHs(mol))}, "
            "so conformer energies cannot be assigned"
        )
    energy_list = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
    for energy_tuple, conf in zip(energy_list, mol.GetConformers()):
        _, energy = energy_tuple
        conf.SetDoubleProp("Energy", energy)
    mol = Chem.RemoveHs(mol)
    return mol


def _symmetry_aware_conformer_rms(mol: Mol, conf_id_1: int, conf_id_2: int) -> float:
    """RMSD between two conformers that accounts for molecular symmetry.

    Signature matches ``AllChem.GetConformerRMS`` so the two are interchangeable.

    :param mol: RDKit molecule holding both conformers
    :param conf_id_1: conformer id used as the reference
    :param conf_id_2: conformer id used as the probe
    :return: the symmetry-corrected RMSD
    """
    return rdMolAlign.GetBestRMS(mol, mol, prbId=conf_id_2, refId=conf_id_1)


def refine_conformers(mol: Mol, energy_threshold: float = 50, rms_threshold: Optional[float] = 0.5,
                      use_symmetry: bool = False) -> Mol:
    """
    Refine the conformers of a molecule by removing those with high energy or low RMSD.

    Conformers more than ``energy_threshold`` above the lowest-energy conformer are
    removed first. The remainder are then reduced so that no two conformers are
    within ``rms_threshold`` of each other, keeping the lowest-energy conformer of
    each redundant group.

    Conformers are addressed by their RDKit conformer id throughout. Ids are not
    required to be contiguous or to start at zero, so a molecule whose conformers
    have already been trimmed (including by an earlier call to this function) is
    handled correctly.

    Note that the conformers are removed from ``mol`` in place; the same molecule
    is returned for convenience. The surviving conformers keep the coordinates they
    came in with: RMSD comparison aligns the conformers it measures, so it is done
    against a throwaway copy rather than against ``mol`` itself.

    :param mol: RDKit molecule with conformers.
    :param energy_threshold: Energy threshold above which conformers are removed.
    :param rms_threshold: RMSD threshold below which conformers are considered redundant and removed.
                          If None, RMSD filtering is skipped.
    :param use_symmetry: If True, compare conformers with a symmetry-aware RMSD
                         (``rdMolAlign.GetBestRMS``) so that conformers differing only by a symmetric
                         relabelling -- a flipped phenyl, a rotated methyl -- are recognised as
                         redundant. The default symmetry-blind comparison reports such a pair as
                         distinct and keeps both. Symmetry-aware comparison has to enumerate the
                         substructure matches, so it is slower, and markedly so for molecules with
                         many symmetric groups.
    :return: RDKit molecule with refined conformers.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have at least one conformer")
    missing = [conf.GetId() for conf in mol.GetConformers() if not conf.HasProp("Energy")]
    if missing:
        raise ValueError(
            f"Conformers {missing} have no 'Energy' property; generate/optimise conformers first (e.g. gen_conformers)"
        )
    # key energies by conformer id, which is not necessarily the position in the list
    energy_dict = {conf.GetId(): float(conf.GetDoubleProp("Energy")) for conf in mol.GetConformers()}
    min_energy = min(energy_dict.values())
    for conf_id, energy in energy_dict.items():
        if energy - min_energy > energy_threshold:
            mol.RemoveConformer(conf_id)

    if rms_threshold is not None:
        # Walk the surviving conformers from lowest to highest energy and keep one
        # only when it differs by at least rms_threshold from every conformer kept
        # so far. Two details matter here:
        #
        #   * Each conformer is compared against the *kept* conformers, not against
        #     all of them. Comparing against all of them discards a conformer for
        #     being close to one that was itself discarded, which prunes conformers
        #     that are not redundant with anything that survives. For conformers
        #     A-B-C where A~B and B~C but A and C are far apart, that would leave
        #     only A; C is a distinct conformer and is kept.
        #   * Low energy first, so the representative kept from each redundant group
        #     is its lowest-energy member rather than whichever happened to be
        #     embedded first.
        rms_func = _symmetry_aware_conformer_rms if use_symmetry else AllChem.GetConformerRMS
        # Both RMSD functions align the conformers they are given, which rototranslates
        # them. Measure on a copy so the caller's coordinates survive; conformer ids are
        # preserved by the copy, so the ids collected here still address mol.
        rms_mol = Chem.Mol(mol)
        conf_ids = sorted((x.GetId() for x in mol.GetConformers()), key=lambda i: energy_dict[i])
        kept_ids = []
        for conf_id in conf_ids:
            if all(rms_func(rms_mol, kept_id, conf_id) >= rms_threshold for kept_id in kept_ids):
                kept_ids.append(conf_id)
        for conf_id in set(conf_ids) - set(kept_ids):
            mol.RemoveConformer(conf_id)
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
    :return: A tuple containing the number of MCS atoms and the RMSD value.
        If the molecules share no common substructure, the RMSD is float("inf").
    """
    mcs_res = FindMCS([mol_1, mol_2])
    num_mcs_atoms = mcs_res.numAtoms
    pat = Chem.MolFromSmarts(mcs_res.smartsString)
    match_1 = mol_1.GetSubstructMatches(pat)
    match_2 = mol_2.GetSubstructMatches(pat)
    min_rmsd = float("inf")
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
        style can be 'line', 'stick', 'sphere', 'cartoon'
    :surface: bool, display SAS
    :opacity: float, opacity of surface, range 0.0-1.0
    :return: viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'cartoon')

    colors = ["lightgray", "pink", "lightgreen", "magenta", "cyan", "orange", "purple"]

    viewer = py3Dmol.view(width=size[0], height=size[1])

    for i, mol in enumerate(mol_list):
        color_idx = i % len(colors)
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
        viewer.setStyle({'model': i}, {style: {'colorscheme': f'{colors[color_idx]}Carbon'}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer


__all__ = [
    "get_center",
    "get_shape_moments",
    "gen_3d",
    "gen_conformers",
    "refine_conformers",
    "get_conformer_energies",
    "mcs_rmsd",
    "mol_to_3D_view",
]
