from typing import List, Tuple, Union

from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from itertools import product


def enumerate_library(rxn_mol: ChemicalReaction, reagent_lol: List[List[Tuple[Mol, str]]]) -> List[
    List[Union[str, str]]]:
    """
    Enumerate a library of products from a given reaction and list of reagents.

    :param rxn_mol: A chemical reaction represented as an RDKit ChemicalReaction object.
    :param reagent_lol: A list of lists, where each inner list represents a set of reagents. Each reagent is a tuple
                        containing an RDKit Mol object and a string identifier.
    :return: A list of lists, where each inner list represents a product. Each product is represented as a list
             containing a SMILES string of the product and a string identifier formed by joining the identifiers of
             the reagents used to form the product.
    """
    prod_list = []
    # itertools.product generates all combinations of reactants
    for reagents in product(*reagent_lol):
        mol_list = [x[0] for x in reagents]
        name_list = [str(x[1]) for x in reagents]
        name = "_".join(name_list)
        prod = rxn_mol.RunReactants(mol_list)
        if prod is not None and len(prod):
            product_mol = prod[0][0]
            Chem.SanitizeMol(product_mol)
            prod_list.append([Chem.MolToSmiles(product_mol), name])
    return prod_list
