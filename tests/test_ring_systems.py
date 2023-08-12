from io import StringIO
import pandas as pd
import useful_rdkit_utils as uru
from tqdm.auto import tqdm


def test_ring_system_finder():
    buff = """SMILES,InChI,Count
c1ccccc1,UHOVQNZJYSORNB-UHFFFAOYSA-N,4
O=c1[nH]c(=O)c2[nH]cnc2[nH]1,LRFVTYWOQMYALW-UHFFFAOYSA-N,2
c1ccncc1,JUJWROOIHBZHMG-UHFFFAOYSA-N,1
O=C1CC(=O)NC(=O)[N-]1,HNYOPLTXPVRDBG-UHFFFAOYSA-M,1
O=C1C=CC(=O)c2ccccc21,FRASJONUBLZVQX-UHFFFAOYSA-N,1
O=C1C=C2CC[C@H]3[C@@H]4CCC[C@H]4CC[C@@H]3[C@H]2CC1,LWNIAOVFYCEEPT-MDBLMMRSSA-N,1
C=C1CCCCC1=C,DYEQHQNRKZJUCT-UHFFFAOYSA-N,1
C=C1CCC[C@@H]2CCC[C@@H]12,UOTNDMDASDMEAG-ZJUUUORDSA-N,1
c1ccc2ccccc2c1,UFWIBTONFRDIAS-UHFFFAOYSA-N,1
c1cncnc1,CZPWVGJYEJSRLH-UHFFFAOYSA-N,1"""
    string_fs = StringIO(buff)
    expected_df = pd.read_csv(string_fs)

    url = "https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/master/data/test.smi"
    uru.create_ring_dictionary(url, "ring_test.csv")
    test_df = pd.read_csv("ring_test.csv")
    pd.testing.assert_frame_equal(test_df, expected_df)


def test_ring_system_lookup():
    url = "https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/master/data/test.smi"
    df = pd.read_csv(url, sep=" ", names=["SMILES", "Name"])
    ring_system_lookup = uru.RingSystemLookup.default()
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
