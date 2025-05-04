import argparse
import json

import pandas as pd
import pystow
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

class SillyWalks:
    def __init__(self) -> None:
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        self.count_dict = {}
        self.load_dict("/Users/pwalters/software/silly_walks/chembl_drugs.smi")

    def build_dict(self, df: pd.DataFrame) -> None:
        """
        Build a dictionary of Morgan fingerprint counts from a DataFrame of SMILES strings.

        :param df: DataFrame containing a column 'SMILES' with SMILES strings
        """
        for smi in df.canonical_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = self.fpgen.GetCountFingerprint(mol)
                for k, v in fp.GetNonzeroElements().items():
                    self.count_dict[k] = self.count_dict.get(k, 0) + v

    def save_dict(self, module: str, name: str) -> None:
        """
        Save the count_dict to disk using pystow.

        :param module: The module name for pystow.
        :param name: The name for the file to save the dictionary.
        """
        pystow.module(module).join(name).write_text(json.dumps(self.count_dict))

    def load_dict(self, filename: str) -> None:
        df = pd.read_csv(filename, sep=" ", names=["canonical_smiles", "name"])
        self.build_dict(df)



    def score(self, smiles_in):
        mol = Chem.MolFromSmiles(smiles_in)
        if mol:
            fp = self.fpgen.GetFingerprint(mol)
            on_bits = fp.GetOnBits()
            silly_bits = [
                x for x in [self.count_dict.get(x) for x in on_bits] if x is None
            ]
            score = len(silly_bits) / len(on_bits)
        else:
            score = 1
        return score

    @staticmethod
    def generate_count_dict(input_file: str, output_file: str) -> None:
        """
        Generate count_dict using the canonical_smiles column in the ChEMBL chemreps file.

        :param input_file: Name of the input file containing the ChEMBL chemreps data.
        :param output_file: Name of the output file where the count_dict will be saved.
        """
        df = pd.read_csv(input_file, sep="\t")
        count_dict = {}
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

        for smi in tqdm(df['canonical_smiles'], desc="Processing SMILES"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = fpgen.GetFingerprint(mol)
                for b in fp.GetOnBits():
                    count_dict[b] = count_dict.get(b, 0) + 1

        with open(output_file, 'w') as f:
            json.dump(count_dict, f)


def main():
    parser = argparse.ArgumentParser(description="Generate count_dict from ChEMBL chemreps file.")
    parser.add_argument("-in", "--input_file", type=str, required=True,
                        help="Name of the input file containing the ChEMBL chemreps data.")
    parser.add_argument("-out", "--output_file", type=str, required=True,
                        help="Name of the output file where the count_dict will be saved.")
    args = parser.parse_args()

    SillyWalks.generate_count_dict(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
