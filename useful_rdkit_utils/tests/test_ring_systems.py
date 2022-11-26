from io import StringIO
import pandas as pd
import useful_rdkit_utils as uru
from tqdm.auto import tqdm


def test_ring_system_finder():
    buff = """,ring_system,count
    0,c1ccccc1,4
    1,O=c1[nH]c(=O)c2[nH]cnc2[nH]1,2
    2,c1ccncc1,1
    3,O=C1CC(=O)NC(=O)[N-]1,1
    4,O=C1C=CC(=O)c2ccccc21,1
    5,O=C1C=C2CC[C@H]3[C@@H]4CCC[C@H]4CC[C@@H]3[C@H]2CC1,1
    6,C=C1CCCCC1=C,1
    7,C=C1CCC[C@@H]2CCC[C@@H]12,1
    8,c1ccc2ccccc2c1,1
    9,c1cncnc1,1
    """
    string_fs = StringIO(buff)
    expected_df = pd.read_csv(string_fs)

    url = "https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/master/data/test.smi"
    uru.create_ring_dictionary(url, "ring_test.csv")
    test_df = pd.read_csv("ring_test.csv")
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_ring_system_lookup():
    url = "https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/master/data/test.smi"
    df = pd.read_csv(url, sep=" ", names=["SMILES", "Name"])
    ring_system_lookup = uru.RingSystemLookup()
    min_freq_list = []
    for smi in tqdm(df.SMILES):
        freq_list = ring_system_lookup.process_smiles(smi)
        if len(freq_list):
            res = min([x[1] for x in freq_list])
        else:
            res = -1
        min_freq_list.append(res)
    df['min_freq'] = min_freq_list
    output_filename = "ring_freq_test.csv"
    df.to_csv(output_filename, index=False)
