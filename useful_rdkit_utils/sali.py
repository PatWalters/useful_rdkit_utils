import pandas as pd
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from rdkit.Chem.Draw import MolsToGridImage
import useful_rdkit_utils as uru
import numpy as np
from tqdm.auto import tqdm

# Constants for fingerprint generation and calculation
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048
SIMILARITY_EPSILON = 0.001  # Small value to avoid division by zero


def calculate_sali(data_frame: pd.DataFrame, smiles_col: str = 'SMILES',
                   activity_col: str = 'Activity') -> pd.DataFrame:
    """
    Calculate the Structure-Activity Landscape Index (SALI) for a dataframe of molecules and activities.

    :param data_frame: DataFrame with columns for SMILES and Activity.
    :param smiles_col: The name of the column containing SMILES strings.
    :param activity_col: The name of the column containing activity values.
    :return: DataFrame containing pairwise SALI values and metadata.
    """
    # Prepare data lists
    smiles_strings = data_frame[smiles_col].tolist()
    activities = data_frame[activity_col].tolist()
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_strings]

    # Generate fingerprints
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_FP_SIZE)
    fingerprints = [fingerprint_generator.GetFingerprint(mol) for mol in molecules]

    sali_records = []
    num_molecules = len(molecules)

    # Calculate total number of pairs for the progress bar: n * (n - 1) / 2
    total_pairs = (num_molecules * (num_molecules - 1)) // 2

    # Calculate pairwise SALI using combinations to avoid redundant comparisons
    progress_bar = tqdm(combinations(range(num_molecules), 2), total=total_pairs, desc="Calculating SALI")
    for i, j in progress_bar:
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])

        delta_activity = abs(activities[i] - activities[j])
        sali_value = delta_activity / (1 - similarity + SIMILARITY_EPSILON)

        sali_records.append({
            'SMILES_1': smiles_strings[i],
            f'{activity_col}_1': activities[i],
            'SMILES_2': smiles_strings[j],
            f'{activity_col}_2': activities[j],
            'Delta_Activity': delta_activity,
            'Tanimoto_Similarity': similarity,
            'SALI': sali_value
        })

    return pd.DataFrame(sali_records)


def plot_sali_pairs(data_frame: pd.DataFrame,
                    smiles_col: str = 'SMILES',
                    activity_col: str = 'Activity',
                    similarity_col: str = 'Tanimoto_Similarity',
                    delta_activity_col: str = 'Delta_Activity',
                    sali_col: str = 'SALI',
                    similarity_cutoff: float = 0.5,
                    delta_activity_cutoff: float = 1.0,
                    mols_per_row: int = 4,
                    pairs_to_show: int = 10) -> object:
    """
    Filters and visualizes molecule pairs from a SALI results DataFrame in a grid.
    Ensures the more active compound of each pair is always displayed on the left.

    :param data_frame: DataFrame containing pairwise SALI results.
    :param smiles_col: Base name of the SMILES column.
    :param activity_col: Base name of the activity column.
    :param similarity_col: Column name for Tanimoto similarity.
    :param delta_activity_col: Column name for the activity difference.
    :param sali_col: Column name for the SALI index.
    :param similarity_cutoff: Minimum similarity threshold for inclusion.
    :param delta_activity_cutoff: Minimum activity difference threshold for inclusion.
    :param mols_per_row: Number of molecules to display per row in the grid image.
    :param pairs_to_show: Maximum number of top SALI pairs to visualize.
    :return: An RDKit grid image showing aligned molecule pairs.
    """
    # Define paired column names
    smi_1, smi_2 = f"{smiles_col}_1", f"{smiles_col}_2"
    act_1, act_2 = f"{activity_col}_1", f"{activity_col}_2"

    # Filter by thresholds, sort by SALI score, and take the top N pairs
    query_filter = f"{delta_activity_col} > {delta_activity_cutoff} and {similarity_col} > {similarity_cutoff}"
    filtered_df = data_frame.query(query_filter).sort_values(sali_col, ascending=False).head(pairs_to_show).copy()

    # Ensure the more active compound is always in the first position (left side)
    needs_swap = filtered_df[act_2] > filtered_df[act_1]
    if needs_swap.any():
        # Swap SMILES
        filtered_df.loc[needs_swap, [smi_1, smi_2]] = filtered_df.loc[needs_swap, [smi_2, smi_1]].values
        # Swap Activities
        filtered_df.loc[needs_swap, [act_1, act_2]] = filtered_df.loc[needs_swap, [act_2, act_1]].values

    # Align molecules to their Maximum Common Substructure (MCS) in pairs
    aligned_mols = []
    for s1, s2 in zip(filtered_df[smi_1], filtered_df[smi_2]):
        aligned_mols.extend(uru.mcs_align([s1, s2]))

    # Flatten activity columns and format as strings for legends
    flattened_activities = filtered_df[[act_1, act_2]].values.flatten()
    activity_legends = [f"{val:.2f}" for val in flattened_activities]

    # Apply global RDKit drawing settings
    uru.rd_make_structures_pretty()

    return MolsToGridImage(aligned_mols, molsPerRow=mols_per_row, legends=activity_legends)
