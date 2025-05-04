from io import StringIO
import pandas as pd
import useful_rdkit_utils as uru
from tqdm.auto import tqdm


def test_ring_system_finder():
    buff = r"""SMILES,InChI,Count
c1ccccc1,UHOVQNZJYSORNB-UHFFFAOYSA-N,12
c1c[nH]cn1,RAXXELZNTBOGNW-UHFFFAOYSA-N,3
C1CCNC1,RWRDLPDLKQPQOW-UHFFFAOYSA-N,2
C1CCOCC1,DHXVGJBLRPWPCS-UHFFFAOYSA-N,2
c1ccoc1,YLQBMQCUIZJEEH-UHFFFAOYSA-N,2
c1cc[nH]c1,KAESVJOAVNADME-UHFFFAOYSA-N,1
O=C1CCCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCCCCCN1,KVCRKKPGTYVCTI-UHFFFAOYSA-N,1
c1cscn1,FZWLAAWBMGSTSO-UHFFFAOYSA-N,1
c1ccc2ccccc2c1,UFWIBTONFRDIAS-UHFFFAOYSA-N,1
O=C1CCOC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CNC(=O)CN1,KZHSFFMCPADVHH-UHFFFAOYSA-N,1
c1cc2ncncc2cn1,PLZDHJUUEGCXJH-UHFFFAOYSA-N,1
C1COCCN1,YNAVUWVOSKDBBP-UHFFFAOYSA-N,1
c1ccc2[nH]ccc2c1,SIKJAQJRHWYJAI-UHFFFAOYSA-N,1
O=C1O[C@H]2/C=C\CCC[C@H]3C=CCC[C@@H]3C[C@@]23O[C@@H]13,UPDVKQFEWFYNLP-FOQDBRFFSA-N,1
"""
    string_fs = StringIO(buff)
    expected_df = pd.read_csv(string_fs)

    chemreps_url = "https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/refs/heads/master/tests/test_chemreps.txt"
    uru.create_ring_dictionary(chemreps_url, "ring_test.csv")
    test_df = pd.read_csv("ring_test.csv")
    test_df.sort_values(["SMILES","InChI"],inplace=True)
    test_df.reset_index(drop=True,inplace=True)
    expected_df.sort_values(["SMILES","InChI"],inplace=True)
    expected_df.reset_index(drop=True,inplace=True)
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
