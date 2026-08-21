import warnings
from typing import List

from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from itertools import product
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



def _sanitized_product_smiles(product_mol: Mol) -> str:
    """Sanitize a reaction product and return its SMILES, or None if it cannot be sanitized.

    ``Chem.SanitizeMol`` raises by default, which aborts an entire enumeration
    because of a single bad product, so errors are caught and reported as a flag.

    :param product_mol: candidate product from RunReactants
    :return: canonical SMILES, or None if the product could not be sanitized
    """
    if Chem.SanitizeMol(product_mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return None
    return Chem.MolToSmiles(product_mol)


def enumerate_library(rxn_mol: ChemicalReaction, reagent_lol: List[List[Mol]]) -> List[List[str]]:
    """
    Enumerate a library of products from a given reaction and list of reagents.

    :param rxn_mol: A chemical reaction represented as an RDKit ChemicalReaction object.
    :param reagent_lol: A list of lists, where each inner list represents a set of reagents. Each reagent is an
    RDKit Mol object. The molecule object must have a "_Name" property that contains a string identifier.
    :return: A list of lists, where each inner list represents a product. Each product is represented as a list
    containing a SMILES string of the product and a string identifier formed by joining the identifiers of
    the reagents used to form the product.
    """
    prod_list = []
    num_failed = 0
    # itertools.product generates all combinations of reactants
    for reagents in product(*reagent_lol):
        mol_list = reagents
        name_list = [x.GetProp("_Name") for x in mol_list]
        name = "_".join(name_list)
        prod = rxn_mol.RunReactants(mol_list)
        if prod is not None and len(prod):
            product_smiles = _sanitized_product_smiles(prod[0][0])
            if product_smiles is None:
                num_failed += 1
                continue
            prod_list.append([product_smiles, name])
    if num_failed:
        warnings.warn(f"{num_failed} product(s) could not be sanitized and were skipped", stacklevel=2)
    return prod_list


def enumerate_library_sample(rxn: ChemicalReaction, reagent_lol: List[List[Mol]], num_to_generate: int) -> pd.DataFrame:
    """
    Enumerate a sample library of products from a given reaction and list of reagents.

    :param rxn: A chemical reaction represented as an RDKit ChemicalReaction object.
    :param reagent_lol: A list of lists, where each inner list represents a set of reagents. Each reagent is an
    RDKit Mol object. The molecule object must have a "_Name" property that contains a string identifier.
    :param num_to_generate: The number of products to generate.
    :return: A pandas DataFrame with the generated products. Each row contains a SMILES string of the product and a
    string identifier.
    """
    used = set()
    prod_list = []
    reagent_counts = [len(x) for x in reagent_lol]
    total_combinations = int(np.prod(reagent_counts))
    with tqdm(total=num_to_generate) as pbar:
        while len(prod_list) < num_to_generate:
            # stop if every reagent combination has already been tried
            if len(used) >= total_combinations:
                break
            # Track combinations by reagent position, not by the joined reagent name.
            # Names are not guaranteed to be unique, and keying on them both skips
            # distinct reagents that happen to share a name and stops len(used) from
            # ever reaching total_combinations, which spins this loop forever.
            reagent_idx = tuple(int(np.random.randint(n)) for n in reagent_counts)
            if reagent_idx in used:
                continue
            used.add(reagent_idx)
            mol_list = [reagent_lol[pos][i] for pos, i in enumerate(reagent_idx)]
            mol_name = "_".join(x.GetProp("_Name") for x in mol_list)
            prod = rxn.RunReactants(mol_list)
            if len(prod):
                product_smiles = _sanitized_product_smiles(prod[0][0])
                if product_smiles is not None:
                    prod_list.append([product_smiles, mol_name])
                    pbar.update(1)
    sample_df = pd.DataFrame(prod_list, columns=["SMILES", "Name"])
    return sample_df


def add_molecule_name(mol_series, name_series):
    for mol, name in zip(mol_series.values, name_series.values):
        mol.SetProp("_Name", str(name))


def reaction_demo():
    """Example of enumerating a library.

    Reads reagent files that are not distributed with the package; run it from a
    directory that has them. Kept as a worked example rather than as public API.
    """
    rxn_smarts = "N[c:4][c:3]C(O)=O.[#6:1][NH2].[#6:2]C(=O)[OH]>>[C:2]c1n[c:4][c:3]c(=O)n1[C:1]"
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    df_list = []
    for filename in ["aminobenzoic", "primary_amines", "carboxylic_acids"]:
        df = pd.read_csv(f"../data/{filename}_100.smi", names=["SMILES", "Name"], sep=" ", header=None)
        df["mol"] = df.SMILES.apply(Chem.MolFromSmiles)
        add_molecule_name(df.mol, df.Name)
        df_list.append(df)
    sample_df = enumerate_library_sample(rxn, [df.mol.values for df in df_list], 1000)
    all_df = enumerate_library(rxn, [df.mol.values[:10] for df in df_list])
    print(len(sample_df), len(all_df))


__all__ = [
    "enumerate_library",
    "enumerate_library_sample",
    "add_molecule_name",
]


if __name__ == "__main__":
    reaction_demo()
