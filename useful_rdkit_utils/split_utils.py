from typing import List
from typing import Union

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.cluster import KMeans

from descriptors import smi2morgan_fp, smi2numpy_fp
from misc_utils import taylor_butina_clustering


def get_scaffold(smi: Union[str, Mol]) -> str:
    """
    Generate the Bemis-Murcko scaffold for a given molecule.

    :param smi: A SMILES string or an RDKit molecule object representing the
                molecule for which to generate the scaffold.
    :return: A SMILES string representing the Bemis-Murcko scaffold of the input
             molecule. If the scaffold cannot be generated, the input SMILES
             string is returned.
    """
    scaffold = MurckoScaffoldSmiles(smi)
    if len(scaffold) == 0:
        scaffold = smi
    return scaffold


def get_butina_clusters(smiles_list: List[str]) -> List[int]:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    :param smiles_list: List of SMILES strings
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    fp_list = [smi2morgan_fp(x) for x in smiles_list]
    return taylor_butina_clustering(fp_list)


def get_bemis_murcko_clusters(smiles_list: List[str]) -> np.ndarray:
    """
    Cluster a list of SMILES strings based on their Bemis-Murcko scaffolds.

    :param smiles_list: List of SMILES strings
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    scaffold_series = pd.Series([get_scaffold(x) for x in smiles_list])
    factorized_values, _ = pd.factorize(scaffold_series)
    return factorized_values


def get_kmeans_clusters(smiles_list: List[str]) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the KMeans clustering algorithm.

    :param smiles_list: List of SMILES strings
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    num_clusters = int(len(smiles_list) / 3.0)
    km = KMeans(n_clusters=num_clusters)
    fp_list = [smi2numpy_fp(x) for x in smiles_list]
    return km.fit_predict(np.stack(fp_list))
