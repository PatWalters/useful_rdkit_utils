import itertools

import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
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


def test_refine_conformers_non_contiguous_conformer_ids(ethanol):
    """Conformer ids need not be contiguous; they must not be treated as list positions."""
    mol = gen_conformers(ethanol, num_confs=10)
    # trim the two lowest ids so the remaining ids start at 2
    mol.RemoveConformer(0)
    mol.RemoveConformer(1)
    remaining_ids = [c.GetId() for c in mol.GetConformers()]
    assert min(remaining_ids) == 2

    refined = refine_conformers(mol, energy_threshold=10, rms_threshold=0.5)
    refined_ids = [c.GetId() for c in refined.GetConformers()]
    assert len(refined_ids) > 0
    # every surviving conformer is one that was actually present beforehand
    assert set(refined_ids).issubset(set(remaining_ids))
    # and every survivor still carries the energy it was refined on
    assert all(c.HasProp("Energy") for c in refined.GetConformers())


def test_refine_conformers_is_repeatable(ethanol):
    """Refining an already refined molecule must not raise."""
    mol = gen_conformers(ethanol, num_confs=10)
    once = refine_conformers(mol, energy_threshold=10, rms_threshold=0.5)
    num_after_once = once.GetNumConformers()
    twice = refine_conformers(once, energy_threshold=10, rms_threshold=0.5)
    assert twice.GetNumConformers() == num_after_once


def test_refine_conformers_requires_a_conformer(ethanol):
    with pytest.raises(ValueError):
        refine_conformers(Chem.MolFromSmiles("CCO"))


def _chain_conformer_mol(deltas, energies):
    """Build a molecule whose conformers differ only by a displacement of atom 0.

    Displacing a single atom by ``delta`` gives an aligned RMSD proportional to
    ``delta``, so the conformers lie on a line in RMSD space and their pairwise
    distances are known in advance. That makes the redundancy filtering testable
    without depending on the vagaries of conformer generation.
    """
    from rdkit.Geometry import Point3D

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    base_pos = np.array(mol.GetConformer(0).GetPositions())
    num_atoms = mol.GetNumAtoms()
    mol.RemoveAllConformers()
    for delta in deltas:
        conf = Chem.Conformer(num_atoms)
        for i, (x, y, z) in enumerate(base_pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        pos = conf.GetAtomPosition(0)
        conf.SetAtomPosition(0, Point3D(pos.x + delta, pos.y, pos.z))
        mol.AddConformer(conf, assignId=True)
    for conf, energy in zip(mol.GetConformers(), energies):
        conf.SetDoubleProp("Energy", energy)
    return mol


def test_refine_conformers_keeps_conformers_that_are_not_redundant():
    """A~B and B~C, but A and C are far apart, so discarding B must not discard C.

    Comparing every conformer against every other one -- rather than against the
    conformers that were kept -- drops C for resembling B, which is itself gone.
    """
    # RMSD is 0.149 between neighbours 0.6 apart and 0.298 between the ends
    mol = _chain_conformer_mol([0.0, 0.6, 1.2], [0.0, 0.0, 0.0])
    refined = refine_conformers(mol, energy_threshold=50, rms_threshold=0.2)
    assert sorted(c.GetId() for c in refined.GetConformers()) == [0, 2]


def test_refine_conformers_keeps_the_lowest_energy_of_a_redundant_group():
    """The survivor of a redundant group is chosen by energy, not by conformer id."""
    # conformer 0 is 10 kcal/mol above conformer 1, and the two are redundant
    mol = _chain_conformer_mol([0.0, 0.15], [10.0, 0.0])
    refined = refine_conformers(mol, energy_threshold=50, rms_threshold=0.2)
    survivors = list(refined.GetConformers())
    assert len(survivors) == 1
    assert survivors[0].GetDoubleProp("Energy") == 0.0


def test_refine_conformers_rms_filtering_invariants():
    """The kept set must be mutually distinct, and nothing distinct may be dropped."""
    deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    rms_threshold = 0.15
    mol = _chain_conformer_mol(deltas, [0.0] * len(deltas))

    # record every pairwise RMSD before anything is removed
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    rms = {(i, j): AllChem.GetConformerRMS(mol, i, j)
           for i, j in itertools.combinations(conf_ids, 2)}

    def lookup(i, j):
        return rms[(i, j)] if (i, j) in rms else rms[(j, i)]

    refined = refine_conformers(mol, energy_threshold=50, rms_threshold=rms_threshold)
    kept = [c.GetId() for c in refined.GetConformers()]
    dropped = [i for i in conf_ids if i not in kept]

    assert len(kept) > 1, "distinct conformers were collapsed into one"
    # 1. nothing redundant survives
    for i, j in itertools.combinations(kept, 2):
        assert lookup(i, j) >= rms_threshold
    # 2. nothing was dropped that was not redundant with a survivor
    for d in dropped:
        assert any(lookup(d, k) < rms_threshold for k in kept)


def test_refine_conformers_rms_threshold_none_skips_filtering():
    deltas = [0.0, 0.05, 0.1]
    mol = _chain_conformer_mol(deltas, [0.0] * len(deltas))
    refined = refine_conformers(mol, energy_threshold=50, rms_threshold=None)
    assert refined.GetNumConformers() == len(deltas)


def _ring_flip_mol():
    """A molecule with two conformers related by a 180 degree phenyl flip.

    The flip maps the ring onto itself, so the two conformers are physically
    identical, but a symmetry-blind RMSD compares atom i to atom i and reports
    them as far apart.
    """
    from rdkit.Chem import rdMolTransforms

    mol = Chem.MolFromSmiles("OCCc1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=0xbeef)
    a, b, c, d = mol.GetSubstructMatch(Chem.MolFromSmarts("[CH2][CH2]c(:c):c"))[:4]
    conf = mol.GetConformer(0)
    start = rdMolTransforms.GetDihedralDeg(conf, a, b, c, d)
    flipped_id = mol.AddConformer(Chem.Conformer(conf), assignId=True)
    rdMolTransforms.SetDihedralDeg(mol.GetConformer(flipped_id), a, b, c, d, start + 180.0)
    for conf, energy in zip(mol.GetConformers(), [0.0, 0.0]):
        conf.SetDoubleProp("Energy", energy)
    return mol


def test_use_symmetry_recognizes_a_symmetry_equivalent_conformer():
    """A flipped phenyl is the same conformer and must collapse to one."""
    refined = refine_conformers(_ring_flip_mol(), energy_threshold=50,
                                rms_threshold=0.5, use_symmetry=True)
    assert refined.GetNumConformers() == 1


def test_default_comparison_is_symmetry_blind():
    """The default is unchanged: the flipped ring is treated as a distinct conformer."""
    refined = refine_conformers(_ring_flip_mol(), energy_threshold=50,
                                rms_threshold=0.5, use_symmetry=False)
    assert refined.GetNumConformers() == 2


def test_use_symmetry_defaults_to_off():
    """Omitting the flag must behave exactly like passing use_symmetry=False."""
    deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    energies = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    implicit = refine_conformers(_chain_conformer_mol(deltas, energies),
                                 energy_threshold=50, rms_threshold=0.15)
    explicit = refine_conformers(_chain_conformer_mol(deltas, energies),
                                 energy_threshold=50, rms_threshold=0.15, use_symmetry=False)
    assert [c.GetId() for c in implicit.GetConformers()] == [c.GetId() for c in explicit.GetConformers()]


def test_use_symmetry_preserves_the_filtering_invariants():
    """Switching metric must not change what correctness means."""
    deltas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    rms_threshold = 0.15
    mol = _chain_conformer_mol(deltas, [0.0] * len(deltas))
    conf_ids = [c.GetId() for c in mol.GetConformers()]
    rms = {(i, j): rdMolAlign.GetBestRMS(mol, mol, prbId=j, refId=i)
           for i, j in itertools.combinations(conf_ids, 2)}

    def lookup(i, j):
        return rms[(i, j)] if (i, j) in rms else rms[(j, i)]

    refined = refine_conformers(mol, energy_threshold=50, rms_threshold=rms_threshold, use_symmetry=True)
    kept = [c.GetId() for c in refined.GetConformers()]
    dropped = [i for i in conf_ids if i not in kept]

    assert len(kept) > 1
    for i, j in itertools.combinations(kept, 2):
        assert lookup(i, j) >= rms_threshold
    for d in dropped:
        assert any(lookup(d, k) < rms_threshold for k in kept)
