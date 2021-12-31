import itertools
from operator import itemgetter

import numpy as np
import py3Dmol
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.Descriptors3D import NPR1, NPR2
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdMolTransforms import ComputeCentroid
from rdkit.ML.Cluster import Butina


# ----------- Molecular geometry
def get_center(mol):
    """
    Get the geometric center of an RDKit molecule
    :param mol: RDKit molecule
    :return: center as a numpy array
    """
    assert mol.GetNumConformers() > 0, "Molecule must have at least one conformer"
    return np.array(ComputeCentroid(mol.GetConformer(0)))


def get_shape_moments(mol):
    """
    Calculate principal moments of inertia as define in https://pubs.acs.org/doi/10.1021/ci025599w
    :param mol: RDKit molecule
    :return: first 2 moments as a tuple
    """
    assert mol.GetNumConformers() > 0, "molecule must have at least one conformer"
    npr1 = NPR1(mol)
    npr2 = NPR2(mol)
    return npr1, npr2


# ----------- Structure cleanup
def count_fragments(mol):
    """
    Count the number of fragments in a molecule
    :param mol: RDKit molecule
    :return: number of fragments
    """
    return len(Chem.GetMolFrags(mol, asMols=True))


def get_largest_fragment(mol):
    """
    Return the fragment with the largest number of atoms
    :param mol: RDKit molecule
    :return: RDKit molecule with the largest number of atoms
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_mw_list.sort(key=itemgetter(0), reverse=True)
    return frag_mw_list[0][1]


# ----------- Descriptors and fingerprints
def mol2morgan_fp(mol, radius=2, nBits=2048):
    """
    Convert an RDKit molecule to a Morgan fingerprint
    :param mol: RDKit molecule
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: RDKit Morgan fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return fp


def mol2numpy_fp(mol, radius=2, nBits=2048):
    """
    Convert an RDKit molecule to a numpy array with Morgan fingerprint bits
    :param mol: RDKit molecule
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: numpy array with RDKit fingerprint bits
    """
    arr = np.zeros((0,), dtype=np.int8)
    fp = mol2morgan_fp(mol=mol, radius=radius, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class RDKitProperties:
    """ Calculate RDKit properties """

    def __init__(self):
        self.property_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        self.property_getter = rdMolDescriptors.Properties(self.property_names)

    def calc_mol(self, mol):
        """
        Calculate properties for a RDKit molecule
        :param mol: RDKit molecule
        :return: a numpy array with properties
        """
        return np.array(self.property_getter.ComputeProperties(mol))

    def calc_smiles(self, smi):
        """
        Calculate properties for a SMILES string
        :param smi: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return np.array(self.calc_mol(mol))
        else:
            return np.array([None] * len(self.property_names))


class Ro5Calculator:
    """
    Calculate Rule of 5 properties + TPSA
    """

    def __init__(self):
        self.names = ["MolWt", "LogP", "HBD", "HBA", "TPSA"]
        self.functions = [MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA]

    def calc_mol(self, mol):
        """
        Calculate properties for a RDKit molecule
        :param mol: RDKit molecule
        :return: a numpy array with properties
        """
        return np.array([x(mol) for x in self.functions])

    def calc_smiles(self, smi):
        """
        Calculate properties for a SMILES string
        :param smi: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return self.calc_mol(mol)
        else:
            return np.array([None] * len(self.functions))


# ----------- Clustering
def taylor_butina_clustering(fp_list, cutoff=0.35):
    """
    Cluster a set of fingerprints using the RDKit Taylor-Buting implementation
    :param fp_list: a list of fingerprints
    :param cutoff: similarity cutoff
    :return: a list of cluster ids
    """
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    cluster_res = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = np.zeros(nfps, dtype=int)
    for cluster_num, cluster in enumerate(cluster_res):
        for member in cluster:
            cluster_id_list[member] = cluster_num
    return cluster_id_list


# ----------- Jupyter
def rd_setup_jupyter():
    """
    Set up rendering the way I want it
    :return: None
    """
    IPythonConsole.ipython_useSVG = True
    IPythonConsole.molSize = 300, 300
    rdDepictor.SetPreferCoordGen(True)


def rd_enable_svg():
    """
    Enable SVG rendering in Jupyter notebooks
    :return: None
    """
    IPythonConsole.ipython_useSVG = True


def rd_enable_png():
    """
    Enable PNG rendering in Jupyter notebooks
    :return: None
    """
    IPythonConsole.ipython_useSVG = False


def rd_set_image_size(x, y):
    """
    Set image size for structure rendering
    :param x: X dimension
    :param y: Y dimension
    :return: None
    """
    IPythonConsole.molSize = x, y


def rd_make_structures_pretty():
    """
    Enable CoordGen rendering
    :return: None
    """
    rdDepictor.SetPreferCoordGen(True)


# ----------- Atom tagging
def label_atoms(mol, labels):
    [atm.SetProp('atomNote', "") for atm in mol.GetAtoms()]
    for atm in mol.GetAtoms():
        idx = atm.GetIdx()
        mol.GetAtomWithIdx(idx).SetProp('atomNote', f"{labels[idx]}")
    return mol


def tag_atoms(mol, atoms_to_tag, tag="x"):
    [atm.SetProp('atomNote', "") for atm in mol.GetAtoms()]
    [mol.GetAtomWithIdx(idx).SetProp('atomNote', tag) for idx in atoms_to_tag]
    return mol


# ----------- Logging
def rd_shut_the_hell_up():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def demo_block_logs():
    from rdkit.rdBase import BlockLogs
    block = BlockLogs()
    # do stuff
    del block


# ----------- Ring stats
def get_spiro_atoms(mol):
    """
    Get atoms that are part of a spiro fusion
    :param mol: input RDKit molecule
    :return: a list of atom numbers for atoms that are the centers of spiro fusions
    """
    info = mol.GetRingInfo()
    ring_sets = [set(x) for x in info.AtomRings()]
    spiro_atoms = []
    for i, j in itertools.combinations(ring_sets, 2):
        i_and_j = (i.intersection(j))
        if len(i_and_j) == 1:
            spiro_atoms += list(i_and_j)
    return spiro_atoms


def max_ring_size(mol):
    """
    Get the size of the largest ring in a molecule
    :param mol: input_molecule
    :return: size of the largest ring or 0 for an acyclic molecule
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) == 0:
        return 0
    else:
        return max([len(x) for x in ri.AtomRings()])


# ----------- 3D related stuff
def gen_3d(mol):
    """
    Generate a 3D structure for a RDKit molecule
    :param mol: input molecule
    :return: molecule with 3D coordinates
    """
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useSmallRingTorsions = True
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, params=params)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        return mol
    except ValueError:
        return None


# from https://birdlet.github.io/2019/10/02/py3dmol_example/
def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')
    mblock = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
