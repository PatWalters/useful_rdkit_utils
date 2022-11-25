#!/usr/bin/env python

import sys
from rdkit import Chem
import pandas as pd
from tqdm.auto import tqdm
from useful_rdkit_utils import RingSystemFinder


class RingSystemLookup:
    def __init__(self, ring_system_csv="chembl_ring_systems.csv"):
        """
        Initialize the lookup table
        :param ring_system_csv: csv file with ring smiles and frequency
        """
        ring_df = pd.read_csv(ring_system_csv)
        self.ring_dict = dict(ring_df[["ring_system", "count"]].values)

    def process_mol(self, mol):
        """
        find ring systems in an RDKit molecule
        :param mol: input molecule
        :return: list of SMILES for ring systems
        """
        if mol:
            ring_system_finder = RingSystemFinder()
            ring_system_list = ring_system_finder.find_ring_systems(mol)
            return [(x, self.ring_dict.get(x) or 0) for x in ring_system_list]
        else:
            return []

    def process_smiles(self, smi):
        """
        find ring systems from a SMILES
        :param smi: input SMILES
        :return: list of SMILES for ring systems
        """
        mol = Chem.MolFromSmiles(smi)
        return self.process_mol(mol)


def test_ring_system_lookup(input_filename, output_filename):
    """
    test for ring system lookup
    :param input_filename: input smiles file
    :param output_filename: output csv file
    :return:
    """
    df = pd.read_csv(input_filename, sep=" ", names=["SMILES", "Name"])
    ring_system_lookup = RingSystemLookup()
    min_freq_list = []
    for smi in tqdm(df.SMILES):
        freq_list = ring_system_lookup.process_smiles(smi)
        if len(freq_list):
            res = min([x[1] for x in freq_list])
        else:
            res = -1
        min_freq_list.append(res)
    df['min_freq'] = min_freq_list
    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    test_ring_system_lookup(sys.argv[1], sys.argv[2])
