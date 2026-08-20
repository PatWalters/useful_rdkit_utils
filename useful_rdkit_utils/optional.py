from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# I wrote the function below for a blog post.  I don't think this is a good strategy for dataset splitting.
# I'm putting the code in optional.py to reduce the dependency burden

def get_t_sne_clusters(smiles_list: List[str], n_clusters: int = 7) -> np.ndarray:
    """
    Cluster a list of SMILES strings using a PCA pre-reduction, a two-dimensional t-SNE
    embedding, and agglomerative clustering.
    From Scaffold Splits Overestimate Virtual Screening Performance
    https://arxiv.org/abs/2406.00873

    :param smiles_list: List of SMILES strings
    :param n_clusters: The number of clusters to use for clustering
    :return: Array of cluster labels corresponding to each SMILES string in the input list.
    """
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fp_list = [fp_gen.GetFingerprintAsNumPy(x) for x in mol_list]
    n = len(mol_list)
    pca = PCA(n_components=min(50, max(1, n - 1)))
    pcs = pca.fit_transform(np.stack(fp_list))
    if n >= 7:
        perplexity = min(30, max(2, (n - 1) // 3))
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(pcs)
    else:
        embedding = PCA(n_components=min(2, max(1, n - 1))).fit_transform(pcs)
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    labels = ac.fit_predict(embedding)
    return labels


# Kept for backward compatibility with the umap-learn based implementation
get_umap_clusters = get_t_sne_clusters


__all__ = [
    "get_t_sne_clusters",
    "get_umap_clusters",
]
