import base64
import io
import sys
from io import StringIO
from operator import itemgetter
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Cluster import Butina
from rdkit.rdBase import BlockLogs
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS


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
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
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


# ----------- Image generation
def boxplot_base64_image(dist: np.ndarray, x_lim: list[int] = [0, 10]) -> str:
    """
    Plot a distribution as a seaborn boxplot and save the resulting image as a base64 image.

    Parameters:
    dist (np.ndarray): The distribution data to plot.
    x_lim (list[int]): The x-axis limits for the boxplot.

    Returns:
    str: The base64 encoded image string.
    """
    sns.set(rc={'figure.figsize': (3, 1)})
    sns.set_style('whitegrid')
    ax = sns.boxplot(x=dist)
    ax.set_xlim(x_lim[0], x_lim[1])
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img align="left" src="data:image/png;base64,%s">' % s


def mol_to_base64_image(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a base64 encoded image string.

    Parameters:
    mol (Chem.Mol): The RDKit molecule to convert.

    Returns:
    str: The base64 encoded image string.
    """
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 150)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    text = drawer.GetDrawingText()
    im_text64 = base64.b64encode(text).decode('utf8')
    img_str = f"<img src='data:image/png;base64, {im_text64}'/>"
    return img_str

# ----------- Align molecules
def remove_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove dummy atoms (atomic number 0) from an RDKit molecule.

    :param mol: RDKit molecule from which to remove dummy atoms.
    :type mol: rdkit.Chem.Mol
    :return: RDKit molecule with dummy atoms removed.
    :rtype: rdkit.Chem.Mol
    """
    rwmol = Chem.RWMol(mol)
    remove_list = [atm.GetIdx() for atm in rwmol.GetAtoms() if atm.GetAtomicNum() == 0]
    for atm_id in sorted(remove_list, reverse=True):
        rwmol.RemoveAtom(atm_id)
    Chem.SanitizeMol(rwmol)
    return Chem.Mol(rwmol)


def align_mols_to_template(template_smiles, smiles_list):
    """
    Aligns a list of molecules to a template molecule defined by its SMILES string.

    Removes dummy atoms (atomic number 0) from the template, generates 2D coordinates,
    and aligns each molecule in the input list to the template structure.

    :param template_smiles: SMILES string of the template molecule.
    :type template_smiles: str
    :param smiles_list: List of SMILES strings to align.
    :type smiles_list: list[str]
    :return: List of aligned RDKit molecule objects.
    :rtype: list[rdkit.Chem.Mol]
    """
    scaffold_mol = (Chem.MolFromSmiles(template_smiles))
    # remove dummy atoms from the scaffold molecule
    scaffold_mol = remove_dummy_atoms(scaffold_mol)
    # generate 2D coordinates for the scaffold molecule
    AllChem.Compute2DCoords(scaffold_mol)
    # generate RDKit molecules for the input structures
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    # generate aligned structures
    [AllChem.GenerateDepictionMatching2DStructure(m,scaffold_mol) for m in mol_list]
    return mol_list

def mcs_align(smiles_list):
    """
    Aligns a list of molecules to their Maximum Common Substructure (MCS).

    Converts SMILES strings to RDKit molecules, finds the MCS (restricted to complete rings),
    generates 2D coordinates for the MCS template, and aligns all molecules to this template.

    :param smiles_list: List of SMILES strings to align.
    :type smiles_list: list[str]
    :return: List of aligned RDKit molecule objects.
    :rtype: list[rdkit.Chem.Mol]
    """
    # convert the SMILES to RDKit molecules
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    # define the parameters for the MCS calculation, if we're aligning it's import that the MCS only contains complete rings
    params = Chem.rdFMCS.MCSParameters()
    params.BondCompareParameters.CompleteRingsOnly=True
    params.AtomCompareParameters.CompleteRingsOnly=True
    # find the MCS
    mcs = FindMCS(mol_list,params)
    # get query molecule from the MCS, we will use this as a template for alignment
    qmol = mcs.queryMol
    # generate coordinates for the template
    AllChem.Compute2DCoords(qmol)
    # generate coordinates for the molecules using the template
    [AllChem.GenerateDepictionMatching2DStructure(m,qmol) for m in mol_list]
    # Draw the molecules, highlighting the MCS
    return mol_list

