from typing import List, Callable, Tuple
from typing import Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from sklearn.cluster import KMeans
from sklearn.model_selection._split import _BaseKFold
from tqdm.auto import tqdm

from .descriptors import smi2morgan_fp, smi2numpy_fp
from .misc_utils import taylor_butina_clustering


class GroupKFoldShuffle(_BaseKFold):
    # from https://github.com/scikit-learn/scikit-learn/issues/20520
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        # Shuffle the unique groups if shuffle is true.
        if self.shuffle:
            np.random.seed(self.random_state)
            unique_groups = np.random.permutation(unique_groups)

        # Split the shuffled groups into n_splits.
        split_groups = np.array_split(unique_groups, self.n_splits)

        # For each split, determine the train and test indices.
        for test_group_ids in split_groups:
            test_mask = np.isin(groups, test_group_ids)
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx


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


def get_random_split(smiles_list: List[str]) -> List[int]:
    """
    Generate a list of integers from 0 to the length of the input list.

    :param smiles_list: A list of SMILES strings.
    :return: A list of integers from 0 to the length of the input list.
    """
    return list(range(0, len(smiles_list)))


def get_butina_clusters(smiles_list: List[str], cutoff: float = 0.65) -> List[int]:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    :param smiles_list: List of SMILES strings
    :param cutoff: The cutoff value to use for clustering
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]
    return taylor_butina_clustering(fp_list, cutoff=cutoff)


def get_bemis_murcko_clusters(smiles_list: List[str]) -> np.ndarray:
    """
    Cluster a list of SMILES strings based on their Bemis-Murcko scaffolds.

    :param smiles_list: List of SMILES strings
    :return: List of cluster labels corresponding to each SMILES string in the input list.
    """
    scaffold_series = pd.Series([get_scaffold(x) for x in smiles_list])
    factorized_values, _ = pd.factorize(scaffold_series)
    return factorized_values


def get_kmeans_clusters(smiles_list: List[str], n_clusters: int = 10) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the KMeans clustering algorithm.

    :param smiles_list: List of SMILES strings
    :param n_clusters: The number of clusters to use for clustering
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    km = KMeans(n_clusters=n_clusters, n_init='auto')
    fp_list = [smi2numpy_fp(x) for x in smiles_list]
    return km.fit_predict(np.stack(fp_list))


def cross_validate(df: pd.DataFrame,
                   model_list: List[Tuple[str, Callable]],
                   y_col: str,
                   descriptor_list: List[Tuple[str, Callable]],
                   group_list: List[Tuple[str, Callable]],
                   metric_list: List[Tuple[str, Callable]],
                   n_outer: int = 5,
                   n_inner: int = 5) -> List[dict]:
    """
    Perform cross-validation on a given dataset.

    :param df: The input pandas DataFrame.
    :param model_list: A list of tuples where each tuple contains a model name and a function to create the model.
    :param y_col: The name of the target column in the DataFrame.
    :param descriptor_list: A list of tuples where each tuple contains a descriptor name and a function to calculate the descriptor.
    :param group_list: A list of tuples where each tuple contains a group name and a function to calculate the group.
    :param metric_list: A list of tuples where each tuple contains a metric name and a function to calculate the metric.
    :param n_outer: The number of outer folds for cross-validation.
    :param n_inner: The number of inner folds for cross-validation.
    :return: A list of dictionaries where each dictionary contains the results for one fold of the cross-validation.
    """
    metric_vals = []
    for i in tqdm(range(0, n_outer), leave=False):
        kf = GroupKFoldShuffle(n_splits=n_inner, shuffle=True)
        for group_name, group_func in group_list:
            current_group = group_func(df.SMILES)
            for j, [train_idx, test_idx] in enumerate(
                    tqdm(kf.split(df, groups=current_group), total=n_inner, leave=False)):
                train = df.iloc[train_idx].copy()
                test = df.iloc[test_idx].copy()
                for descriptor_name, descriptor_func in descriptor_list:
                    descriptor_func(descriptor_name, train, test)
                    for model_name, model_func in model_list:
                        model = model_func()
                        model.fit(np.stack(train[descriptor_name]), train[y_col])
                        pred = model.predict(np.stack(test[descriptor_name]))
                        fold = i * n_outer + j
                        metric_dict = {'group': group_name, 'model': model_name, 'desc': descriptor_name, 'fold': fold}
                        for metric_name, metric_func in metric_list:
                            metric_dict[metric_name] = metric_func(test[y_col], pred)
                        metric_vals.append(metric_dict)
    return metric_vals
