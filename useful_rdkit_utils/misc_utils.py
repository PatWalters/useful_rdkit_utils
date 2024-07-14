import sys
from io import StringIO
from operator import itemgetter
from typing import List
from typing import Tuple

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs


# ----------- Structure reading and cleanup

def smi2mol_with_errors(smi: str) -> Tuple[Mol, str]:
    """ Parse SMILES and return any associated errors or warnings

    :param smi: input SMILES
    :return: tuple of RDKit molecule, warning or error
    """
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromSmiles(smi)
    err = sio.getvalue()
    sio = sys.stderr = StringIO()
    sys.stderr = sys.__stderr__
    return mol, err


def count_fragments(mol: Mol) -> int:
    """Count the number of fragments in a molecule

    :param mol: RDKit molecule
    :return: number of fragments
    """
    return len(Chem.GetMolFrags(mol, asMols=True))


def get_largest_fragment(mol: Mol) -> Mol:
    """Return the fragment with the largest number of atoms

    :param mol: RDKit molecule
    :return: RDKit molecule with the largest number of atoms
    """
    frag_list = list(Chem.GetMolFrags(mol, asMols=True))
    frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
    frag_mw_list.sort(key=itemgetter(0), reverse=True)
    return frag_mw_list[0][1]


# ----------- Clustering
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
def taylor_butina_clustering(fp_list: List[DataStructs.ExplicitBitVect], cutoff: float = 0.65) -> List[int]:
    """Cluster a set of fingerprints using the RDKit Taylor-Butina implementation

    :param fp_list: a list of fingerprints
    :param cutoff: distance cutoff (1 - Tanimoto similarity)
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
    return cluster_id_list.tolist()


# ----------- Atom tagging
def label_atoms(mol: Mol, labels: List[str]) -> Mol:
    """Label atoms when depicting a molecule

    :param mol: input molecule
    :param labels: labels, one for each atom
    :return: molecule with labels
    """
    [atm.SetProp('atomNote', "") for atm in mol.GetAtoms()]
    for atm in mol.GetAtoms():
        idx = atm.GetIdx()
        mol.GetAtomWithIdx(idx).SetProp('atomNote', f"{labels[idx]}")
    return mol


def tag_atoms(mol: Mol, atoms_to_tag: List[int], tag: str = "x") -> Mol:
    """Tag atoms with a specified string

    :param mol: input molecule
    :param atoms_to_tag: indices of atoms to tag
    :param tag: string to use for the tags
    :return: molecule with atoms tagged
    """
    [atm.SetProp('atomNote', "") for atm in mol.GetAtoms()]
    [mol.GetAtomWithIdx(idx).SetProp('atomNote', tag) for idx in atoms_to_tag]
    return mol


# ----------- Logging
def rd_shut_the_hell_up() -> None:
    """Make the RDKit be a bit more quiet

    :return: None
    """
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)


def demo_block_logs() -> None:
    """An example of another way to turn off RDKit logging

    :return: None
    """
    block = BlockLogs()
    # do stuff
    del block
