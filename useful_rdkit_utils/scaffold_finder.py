import itertools
import sys
from glob import glob
from typing import Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
from rdkit.Chem.rdchem import Mol
from tqdm.auto import tqdm

from .misc_utils import get_largest_fragment


def cleanup_fragment(mol: Mol) -> Tuple[Mol, int]:
    """
    Replace atom map numbers with Hydrogens
    :param mol: input molecule
    :return: modified molecule, number of R-groups
    """
    rgroup_count = 0
    for atm in mol.GetAtoms():
        atm.SetAtomMapNum(0)
        if atm.GetAtomicNum() == 0:
            rgroup_count += 1
            atm.SetAtomicNum(1)
    mol = Chem.RemoveAllHs(mol)
    return mol, rgroup_count


def generate_fragments(mol: Mol) -> pd.DataFrame:
    """
    Generate fragments using the RDKit
    :param mol: RDKit molecule
    :raises ValueError: if mol is None or contains no atoms
    :return: a Pandas dataframe with Scaffold SMILES, Number of Atoms, Number of R-Groups
    """
    if mol is None:
        raise ValueError("Cannot generate fragments for None")
    num_mol_atoms = mol.GetNumAtoms()
    if num_mol_atoms == 0:
        raise ValueError("Cannot generate fragments for a molecule with no atoms")
    # Generate molecule fragments
    frag_list = FragmentMol(mol)
    # Flatten the output into a single list
    flat_frag_list = [x for x in itertools.chain(*frag_list) if x]
    # The output of Fragment mol is contained in single molecules.  Extract the largest fragment from each molecule
    flat_frag_list = [get_largest_fragment(x) for x in flat_frag_list]
    # Keep fragments where the number of atoms in the fragment is at least 2/3 of the number of atoms
    # in the input molecule
    flat_frag_list = [x for x in flat_frag_list if x.GetNumAtoms() / num_mol_atoms > 0.67]
    # remove atom map numbers from the fragments
    flat_frag_list = [cleanup_fragment(x) for x in flat_frag_list]
    # Convert fragments to SMILES
    frag_smiles_list = [[Chem.MolToSmiles(x), x.GetNumAtoms(), y] for (x, y) in flat_frag_list]
    # Add the input molecule to the fragment list
    frag_smiles_list.append([Chem.MolToSmiles(mol), mol.GetNumAtoms(), 1])
    # Put the results into a Pandas dataframe
    frag_df = pd.DataFrame(frag_smiles_list, columns=["Scaffold", "NumAtoms", "NumRgroups"])
    # Remove duplicate fragments
    frag_df = frag_df.drop_duplicates("Scaffold")
    return frag_df


def find_scaffolds(df_in: pd.DataFrame, smiles_col="SMILES", name_col="Name",disable_progress=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate scaffolds for a set of molecules
    :param df_in: Pandas dataframe with SMILES, Name and "mol" (RDKit molecule) columns
    :param smiles_col: name of the SMILES column
    :param name_col: name of the name column
    :param disable_progress: hide the progress bar
    :raises KeyError: if any of the required columns is missing
    :return: dataframe with molecules and scaffolds, dataframe with unique scaffolds
    """
    missing = [c for c in (smiles_col, name_col, "mol") if c not in df_in.columns]
    if missing:
        raise KeyError(
            f"find_scaffolds requires the column(s) {missing}. The 'mol' column holds RDKit "
            f"molecules, e.g. df['mol'] = df['{smiles_col}'].apply(Chem.MolFromSmiles)"
        )
    # Loop over molecules and generate fragments, fragments for each molecule are returned as a Pandas dataframe
    df_list = []
    for smiles, name, mol in tqdm(df_in[[smiles_col, name_col, "mol"]].values, disable=disable_progress):
        tmp_df = generate_fragments(mol).copy()
        tmp_df['Name'] = name
        tmp_df['SMILES'] = smiles
        df_list.append(tmp_df)
    # Combine the list of dataframes into a single dataframe
    mol_df = pd.concat(df_list)
    # Collect scaffolds
    scaffold_list = []
    for k, v in mol_df.groupby("Scaffold"):
        scaffold_list.append([k, len(v.Name.unique()), v.NumAtoms.values[0]])
    scaffold_df = pd.DataFrame(scaffold_list, columns=["Scaffold", "Count", "NumAtoms"])
    # Any fragment that occurs more times than the number of fragments can't be a scaffold
    num_df_rows = len(df_in)  # noqa: F841  (referenced by name inside the query below)
    scaffold_df = scaffold_df.query("Count <= @num_df_rows")
    # Sort scaffolds by frequency
    scaffold_df = scaffold_df.sort_values(["Count", "NumAtoms"], ascending=[False, False])
    return mol_df, scaffold_df


def get_molecules_with_scaffold(
    scaffold: str,
    mol_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    smiles_col: str = "SMILES",
    name_col: str = "Name",
    activity_col: str = "pIC50",
    extra_cols: List[str] = []
) -> Tuple[List[str], pd.DataFrame]:
    """
    Find molecules in a DataFrame that match a given scaffold and return their core structures and activity data.

    :param scaffold: SMILES string of the scaffold to match.
    :param mol_df: DataFrame containing molecule information and fragments.
    :param activity_df: DataFrame containing activity data for molecules.
    :param smiles_col: Name of the column containing SMILES strings.
    :param name_col: Name of the column containing molecule names.
    :param activity_col: Name of the column containing activity values.
    :param extra_cols: Additional columns to include in the output DataFrame.
    :return: A tuple with a list of unique core SMILES and a DataFrame of matching molecules with activity data.
    """
    match_df = mol_df.query("Scaffold == @scaffold").copy()
    # Build the molecules from the fragment table's own SMILES. Taking them from a
    # "mol" column in activity_df instead leaves NaN for any scaffold match that has
    # no activity row, and RGroupDecompose then fails with an opaque Boost error.
    match_df['mol'] = match_df.SMILES.apply(Chem.MolFromSmiles)
    merge_df = match_df.merge(activity_df, left_on=["SMILES", "Name"],
                              right_on=[smiles_col, name_col], how="left",
                              suffixes=("", "_activity"))
    scaffold_mol = Chem.MolFromSmiles(scaffold)
    Chem.RemoveStereochemistry(scaffold_mol)
    rgroup_match, rgroup_miss = RGroupDecompose(
        scaffold_mol, merge_df.mol, asSmiles=True
    )
    output_cols = [smiles_col, name_col, activity_col] + extra_cols
    if len(rgroup_match):
        rgroup_df = pd.DataFrame(rgroup_match)
        return rgroup_df.Core.unique(), merge_df[output_cols]
    else:
        return [], merge_df[output_cols]


def main():
    """
    Read all SMILES files in the current directory, generate scaffolds, report stats for each scaffold
    :return: None
    """
    filename_list = glob("CHEMBL237.smi")
    out_list = []
    for filename in filename_list:
        print(filename)
        input_df = pd.read_csv(filename, names=["SMILES", "Name", "pIC50"])
        input_df['mol'] = input_df.SMILES.apply(Chem.MolFromSmiles)
        input_df['SMILES'] = input_df.mol.apply(Chem.MolToSmiles)
        mol_df, scaffold_df = find_scaffolds(input_df)
        scaffold_1 = scaffold_df.Scaffold.values[0]
        scaffold_list, scaffold_mol_df = get_molecules_with_scaffold(scaffold_1, mol_df, input_df)
        ic50_list = scaffold_mol_df.pIC50
        if len(scaffold_list):
            print(scaffold_list[0], len(scaffold_mol_df), max(ic50_list) - min(ic50_list), np.std(ic50_list))
            out_list.append([filename, scaffold_list[0], len(scaffold_mol_df), max(ic50_list) - min(ic50_list),
                             np.std(ic50_list)])
        else:
            print(f"Could not find a scaffold for {filename}", file=sys.stderr)
    out_df = pd.DataFrame(out_list, columns=["Filename", "Scaffold", "Count", "Range", "Std"])
    out_df.to_csv("scaffold_stats.csv", index=False)


__all__ = [
    "cleanup_fragment",
    "generate_fragments",
    "find_scaffolds",
    "get_molecules_with_scaffold",
]


if __name__ == "__main__":
    main()
