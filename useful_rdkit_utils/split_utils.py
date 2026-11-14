from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.cluster import KMeans
from sklearn.model_selection._split import _BaseKFold
from tqdm.auto import tqdm
from .descriptors import smi2numpy_fp
from .misc_utils import get_largest_fragment, taylor_butina_clustering


class GroupKFoldShuffle(_BaseKFold):
    # from https://github.com/scikit-learn/scikit-learn/issues/20520
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        # Find the unique groups in the dataset.
        unique_groups = np.unique(groups)

        if self.n_splits > len(unique_groups):
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than the number of groups="
                f"{len(unique_groups)}"
            )

        # Shuffle the unique groups if shuffle is true.
        if self.shuffle:
            random_state = np.random.RandomState(self.random_state)
            unique_groups = random_state.permutation(unique_groups)

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
    if isinstance(smi, str):
        scaffold = MurckoScaffoldSmiles(smi)
    else:
        scaffold = MurckoScaffoldSmiles(mol=smi)
    if not scaffold:
        scaffold = Chem.MolToSmiles(smi) if isinstance(smi, Mol) else smi
    return scaffold


def get_random_clusters(smiles_list: List[str]) -> np.ndarray:
    """
    Assign every SMILES to its own unique group.

    This makes a grouped splitter (e.g. GroupKFoldShuffle) behave like a
    random (non-grouped) split, since no two molecules are forced into the
    same fold.

    :param smiles_list: A list of SMILES strings.
    :return: Array of integers from 0 to the length of the input list.
    """
    return np.arange(len(smiles_list))





def get_butina_clusters(smiles_list: List[str], dist_cutoff: float = 0.65) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the Butina clustering algorithm.

    :param smiles_list: List of SMILES strings
    :param dist_cutoff: distance cutoff (1 - Tanimoto similarity), see
        :func:`useful_rdkit_utils.taylor_butina_clustering`
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_list = [fg.GetFingerprint(x) for x in mol_list]
    return np.asarray(taylor_butina_clustering(fp_list, dist_cutoff=dist_cutoff))


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
                   model_list: List[Tuple[str, Callable[[str], object]]],
                   y_col: str,
                   group_list: List[Tuple[str, Callable[[pd.Series], pd.Series]]],
                   n_outer: int = 5,
                   n_inner: int = 5,
                   random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Perform nested cross-validation on a dataset using multiple models and grouping strategies.

    For each grouping strategy the data is split into n_outer outer folds.
    For every outer fold the inner models are cross-validated on the outer
    TRAINING set only (group assignments are computed there as well, to avoid
    leaking information from the held-out fold). The models are then refit on
    the full outer training set and evaluated on the outer test fold.

    The resulting ``fold`` column uses the label ``outer * (n_inner + 1) + inner``
    for the inner splits; the outer test evaluation occupies slot ``outer * (n_inner + 1)
    + n_inner`` so that inner and outer metrics never collide.

    :param df: The input dataframe containing the data.
    :param model_list: A list of tuples where each tuple contains a model name and a class that, when called
        with ``y_col``, returns a model with a ``validate(train, test)`` method.
    :param y_col: The name of the target column.
    :param group_list: A list of tuples where each tuple contains a group name and a callable that assigns
        groups based on the SMILES column.
    :param n_outer: The number of outer folds for cross-validation. Default is 5.
    :param n_inner: The number of inner folds for cross-validation. Default is 5.
    :param random_state: Seed for the shuffled splits. Default is None.
    :return: A dataframe containing predictions (and model names as columns) for each fold, model, and group.
    """
    fold_df_list = []
    input_cols = df.columns
    for group_name, group_func in group_list:
        # assign groups based on cluster, scaffold, etc
        # group_func is user supplied and may return a list, so normalise to an array:
        # pd.unique() on a plain list is deprecated and will raise in a future pandas
        outer_group = np.asarray(group_func(df.SMILES))
        outer_splits = min(n_outer, len(pd.unique(outer_group)))
        outer_kf = GroupKFoldShuffle(n_splits=outer_splits, shuffle=True, random_state=random_state)
        for i, [outer_train_idx, outer_test_idx] in enumerate(
                tqdm(outer_kf.split(df, groups=outer_group), total=outer_splits, desc=group_name, leave=False)):
            outer_train = df.iloc[outer_train_idx].copy()
            outer_test = df.iloc[outer_test_idx].copy()
            inner_group = np.asarray(group_func(outer_train.SMILES))
            inner_splits = min(n_inner, len(pd.unique(inner_group)))
            inner_random_state = None if random_state is None else random_state + 1000 * (i + 1)
            inner_kf = GroupKFoldShuffle(n_splits=inner_splits, shuffle=True, random_state=inner_random_state)
            for j, [train_idx, test_idx] in enumerate(
                    tqdm(inner_kf.split(outer_train, groups=inner_group), total=inner_splits,
                         desc=f"{group_name} inner", leave=False)):
                fold = i * (n_inner + 1) + j
                train = outer_train.iloc[train_idx].copy()
                test = outer_train.iloc[test_idx].copy()

                train['dset'] = 'train'
                test['dset'] = 'test'
                train['group'] = group_name
                test['group'] = group_name
                train['fold'] = fold
                test['fold'] = fold

                for model_name, model_class in model_list:
                    model = model_class(y_col)
                    pred = model.validate(train, test)
                    test[model_name] = pred
                fold_df_list.append(pd.concat([train, test]))

            # refit on the full outer training set and evaluate on the outer test fold
            fold = i * (n_inner + 1) + n_inner
            train = outer_train.copy()
            test = outer_test.copy()
            train['dset'] = 'train'
            test['dset'] = 'test'
            train['group'] = group_name
            test['group'] = group_name
            train['fold'] = fold
            test['fold'] = fold
            for model_name, model_class in model_list:
                model = model_class(y_col)
                pred = model.validate(train, test)
                test[model_name] = pred
            fold_df_list.append(test)
    output_cols = list(input_cols) + ['dset', 'group', 'fold'] + [x[0] for x in model_list]
    return pd.concat(fold_df_list)[output_cols]


def _leakage_keys(smiles_list: List[str], strip_salts: bool,
                 progress: bool) -> Tuple[List[Optional[str]], List[Optional[str]], List[Optional[Mol]]]:
    """InChIKey, Bemis-Murcko scaffold and parsed molecule for each input SMILES.

    :param smiles_list: SMILES to key
    :param strip_salts: keep only the largest fragment before keying
    :param progress: show a progress bar
    :return: a list of InChIKeys, a list of scaffolds and a list of molecules,
        each holding None where the SMILES would not parse
    """
    keys, scaffolds, mols = [], [], []
    for smi in tqdm(smiles_list, disable=not progress, desc="keying molecules"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            keys.append(None)
            scaffolds.append(None)
            mols.append(None)
            continue
        if strip_salts:
            mol = get_largest_fragment(mol)
        keys.append(Chem.MolToInchiKey(mol))
        scaffolds.append(get_scaffold(mol))
        mols.append(mol)
    return keys, scaffolds, mols


def audit_split_leakage(train_smiles: List[str], test_smiles: List[str],
                        similarity_cutoff: float = 0.9, strip_salts: bool = True,
                        radius: int = 2, n_bits: int = 2048,
                        progress: bool = False) -> pd.DataFrame:
    """
    Check an existing train/test split for test molecules that are already in the training set.

    This module builds splits; this function audits one it is handed, which is the case when the
    split arrives with a public benchmark or from a collaborator. Each test molecule is checked
    against the training set four ways, from strictest to loosest:

    * ``exact``, an identical InChIKey. The same molecule on both sides.
    * ``same_connectivity``, an identical first InChIKey block. That block encodes constitution
      only, so it also catches a training molecule that differs just in stereochemistry or
      protonation state: enantiomers, diastereomers, E/Z isomers and an acid against its
      conjugate base all collapse onto it. It does not see through every tautomer, and it does
      not see through salts and mixtures at all, which is what ``strip_salts`` is for.
    * ``same_scaffold``, an identical Bemis-Murcko scaffold, via :func:`get_scaffold`.
    * ``nearest_similarity``, the largest Morgan fingerprint Tanimoto to any training molecule,
      with the training molecule that produced it. ``near_duplicate`` flags the ones at or above
      ``similarity_cutoff``. The fingerprint carries no chirality, so a pair of enantiomers
      scores 1.0 here; that pair is already reported by ``same_connectivity``.

    The first two are leakage outright: a model can score well on them without generalizing at
    all. The last two are degrees of optimism rather than errors, and whether they matter depends
    on what the split was built to measure, so they are reported rather than judged.

    :param train_smiles: SMILES of the training set
    :param test_smiles: SMILES of the test set, one row of output per entry
    :param similarity_cutoff: Tanimoto at or above which a test molecule is flagged a near
                              duplicate of a training molecule
    :param strip_salts: Reduce every molecule to its largest fragment before comparing, so a
                        salt of a training compound is recognized rather than missed. Counter
                        ions and solvates otherwise change the InChIKey outright.
    :param radius: Morgan fingerprint radius used for the similarity column
    :param n_bits: Morgan fingerprint length used for the similarity column
    :param progress: Show progress bars, which is worth it on large training sets
    :return: One row per test molecule, with columns ``test_smiles``, ``exact``,
        ``same_connectivity``, ``same_scaffold``, ``nearest_similarity``, ``nearest_train_smiles``
        and ``near_duplicate``. A test SMILES that will not parse yields False flags, a NaN
        similarity and None for the nearest training molecule.
    """
    train_keys, train_scaffolds, train_mols = _leakage_keys(list(train_smiles), strip_salts, progress)
    test_keys, test_scaffolds, test_mols = _leakage_keys(list(test_smiles), strip_salts, progress)

    exact_lookup, connectivity_lookup, scaffold_set = {}, {}, set()
    for smi, key, scaffold in zip(train_smiles, train_keys, train_scaffolds):
        if key is None:
            continue
        exact_lookup.setdefault(key, smi)
        connectivity_lookup.setdefault(key.split("-")[0], smi)
        scaffold_set.add(scaffold)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    train_fp_list = [generator.GetFingerprint(mol) for mol in train_mols if mol is not None]
    train_fp_smiles = [smi for smi, mol in zip(train_smiles, train_mols) if mol is not None]

    rows = []
    for smi, key, scaffold, mol in tqdm(list(zip(test_smiles, test_keys, test_scaffolds, test_mols)),
                                        disable=not progress, desc="auditing test set"):
        nearest_similarity, nearest_smiles = np.nan, None
        if mol is not None and train_fp_list:
            similarity = np.array(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol),
                                                                     train_fp_list))
            best = int(similarity.argmax())
            nearest_similarity, nearest_smiles = float(similarity[best]), train_fp_smiles[best]
        rows.append({
            "test_smiles": smi,
            "exact": key is not None and key in exact_lookup,
            "same_connectivity": key is not None and key.split("-")[0] in connectivity_lookup,
            "same_scaffold": scaffold is not None and scaffold in scaffold_set,
            "nearest_similarity": nearest_similarity,
            "nearest_train_smiles": nearest_smiles,
            "near_duplicate": bool(nearest_similarity >= similarity_cutoff)
            if nearest_similarity == nearest_similarity else False,
        })
    audit_df = pd.DataFrame(rows)
    # A column of strings with a gap in it becomes NaN on construction. Put the None back so
    # the absence of a nearest neighbor reads the same whatever else the column holds.
    nearest = audit_df["nearest_train_smiles"]
    audit_df["nearest_train_smiles"] = nearest.astype(object).where(nearest.notna(), None)
    return audit_df


def summarize_split_leakage(audit_df: pd.DataFrame) -> pd.Series:
    """
    Reduce the output of :func:`audit_split_leakage` to the fraction of the test set each check hit.

    :param audit_df: the DataFrame returned by :func:`audit_split_leakage`
    :return: a Series indexed by check name, holding the fraction of test rows each one flagged,
        plus ``n_test`` and the median nearest-neighbor similarity
    """
    checks = ["exact", "same_connectivity", "same_scaffold", "near_duplicate"]
    summary = {name: float(audit_df[name].mean()) for name in checks}
    summary["median_nearest_similarity"] = float(audit_df["nearest_similarity"].median())
    summary["n_test"] = int(len(audit_df))
    return pd.Series(summary)


__all__ = [
    "GroupKFoldShuffle",
    "get_scaffold",
    "get_random_clusters",
    "get_butina_clusters",
    "get_bemis_murcko_clusters",
    "get_kmeans_clusters",
    "cross_validate",
    "audit_split_leakage",
    "summarize_split_leakage",
]
