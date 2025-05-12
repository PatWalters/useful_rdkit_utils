from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
try:
    from umap import UMAP
except ImportError:
    pass


# I wrote the function below for a blog post.  I don't think this is a good strategy for dataset splitting.
# I'm putting the code in optional.py to reduce the dependency burden

def get_umap_clusters(smiles_list: List[str], n_clusters: int = 7) -> np.ndarray:
    """
    Cluster a list of SMILES strings using the umap clustering algorithm.
    From Scaffold Splits Overestimate Virtual Screening Performance
    https://arxiv.org/abs/2406.00873

    :param smiles_list: List of SMILES strings
    :param n_clusters: The number of clusters to use for clustering
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fp_list = [fp_gen.GetFingerprintAsNumPy(x) for x in mol_list]
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(np.stack(fp_list))
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(pcs)
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac.fit_predict(embedding)
    return ac.labels_