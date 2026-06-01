import pandas as pd
import pytest
import useful_rdkit_utils as uru


@pytest.fixture
def small_dataset():
    return pd.DataFrame({
        "SMILES": [
            "c1ccccc1",
            "c1ccccc1C",
            "c1ccccc1CC",
            "CCCCCCCC",
            "O=C(O)c1ccccc1",
        ],
        "Activity": [5.0, 5.2, 5.5, 1.0, 7.5],
    })


def test_calculate_sali_shape_and_columns(small_dataset):
    result = uru.calculate_sali(small_dataset)
    expected_pairs = len(small_dataset) * (len(small_dataset) - 1) // 2
    assert len(result) == expected_pairs
    for col in ["SMILES_1", "SMILES_2", "Activity_1", "Activity_2",
                "Delta_Activity", "Tanimoto_Similarity", "SALI"]:
        assert col in result.columns


def test_calculate_sali_values(small_dataset):
    result = uru.calculate_sali(small_dataset)
    assert (result["Delta_Activity"] >= 0).all()
    assert (result["Tanimoto_Similarity"] >= 0).all()
    assert (result["Tanimoto_Similarity"] <= 1).all()
    assert (result["SALI"] >= 0).all()


def test_calculate_sali_custom_column_names():
    df = pd.DataFrame({
        "smi": ["c1ccccc1", "c1ccccc1C", "CCCC"],
        "pIC50": [6.0, 6.5, 4.0],
    })
    result = uru.calculate_sali(df, smiles_col="smi", activity_col="pIC50")
    assert "pIC50_1" in result.columns
    assert "pIC50_2" in result.columns
    assert len(result) == 3


def test_plot_sali_pairs_returns_image(small_dataset):
    sali_df = uru.calculate_sali(small_dataset)
    img = uru.plot_sali_pairs(
        sali_df,
        similarity_cutoff=0.0,
        delta_activity_cutoff=0.0,
        pairs_to_show=3,
    )
    assert img is not None


def test_plot_sali_pairs_with_swap(small_dataset):
    """Force a swap by including pairs where Activity_2 > Activity_1."""
    sali_df = uru.calculate_sali(small_dataset)
    img = uru.plot_sali_pairs(
        sali_df,
        similarity_cutoff=0.0,
        delta_activity_cutoff=0.0,
        mols_per_row=2,
        pairs_to_show=5,
    )
    assert img is not None
