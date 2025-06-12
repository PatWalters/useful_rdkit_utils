from typing import List, Union

from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from itertools import product
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



def enumerate_library(rxn_mol: ChemicalReaction, reagent_lol: List[List[Mol]]) -> List[
    List[Union[str, str]]]:
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
    # itertools.product generates all combinations of reactants
    for reagents in product(*reagent_lol):
        mol_list = reagents
        name_list = [x.GetProp("_Name") for x in mol_list]
        name = "_".join(name_list)
        prod = rxn_mol.RunReactants(mol_list)
        if prod is not None and len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            prod_list.append([Chem.MolToSmiles(product_mol), name])
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
    count = 0
    with tqdm(total=num_to_generate) as pbar:
        while True:
            mol_list = [np.random.choice(x) for x in reagent_lol]
            name_list = [x.GetProp("_Name") for x in mol_list]
            mol_name = "_".join(name_list)
            if mol_name in used:
                continue
            used.add(mol_name)
            prod = rxn.RunReactants(mol_list)
            if len(prod):
                prod_mol = prod[0][0]
                res = Chem.SanitizeMol(prod_mol)
                if res == Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                    prod_list.append([Chem.MolToSmiles(prod_mol), mol_name])
                    count += 1
                if count % 100 == 0:
                    pbar.update(100)
            if count >= num_to_generate:
                break
    sample_df = pd.DataFrame(prod_list, columns=["SMILES", "Name"])
    return sample_df


def add_molecule_name(mol_series, name_series):
    for mol, name in zip(mol_series.values, name_series.values):
        mol.SetProp("_Name", str(name))


def reaction_demo():
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


if __name__ == "__main__":
    reaction_demo()
