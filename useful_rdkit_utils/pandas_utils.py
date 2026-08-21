import pandas as pd
import useful_rdkit_utils

# from  https://stackoverflow.com/questions/47136436/python-pandas-convert-value-counts-output-to-dataframe


def value_counts_df(df_in, col_in):
    """Returns pd.value_counts() as a DataFrame

    :param df_in: Dataframe on which to run value_counts(), must have column `col`.
    :param col_in: Name of column in `df` for which to generate counts
    :return: Returned dataframe will have two columns, one named "count" which contains the count_values()
        for each unique value of df[col]. The other column will be named `col`.
    """
    df_out = pd.DataFrame(df_in[col_in].value_counts())
    df_out.index.name = col_in
    df_out.columns = ['count']
    return df_out.reset_index()


def add_molecule_and_errors(df_in, smiles_col='SMILES', mol_col_name='ROMol', error_col_name="Error"):
    """Add a molecule column and another column with associated errors to a Pandas dataframe

    :param df_in: input dataframe
    :param smiles_col: name for the input SMILES column
    :param mol_col_name: name for the output molecule column
    :param error_col_name: name for the output errors column
    :return: None
    """
    df_in[[mol_col_name, error_col_name]] = df_in[smiles_col].apply(useful_rdkit_utils.smi2mol_with_errors).to_list()


def split_dataframe(df, chunk_size=10000):
    """Split a dataframe into chunks of at most chunk_size rows

    :param df: dataframe to split
    :param chunk_size: maximum number of rows per chunk
    :return: list of dataframes, none of them empty
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    # ceiling division; a plain "// chunk_size + 1" appends an empty chunk whenever
    # len(df) is an exact multiple of chunk_size
    num_chunks = -(-len(df) // chunk_size)
    return [df[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


def get_dataframe_nans(df):
    return df[df.isnull().any(axis=1)]


__all__ = [
    "value_counts_df",
    "add_molecule_and_errors",
    "split_dataframe",
    "get_dataframe_nans",
]