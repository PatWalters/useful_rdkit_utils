import argparse
import json

import pandas as pd
import pystow
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm


class SillyWalks:
    def __init__(self, dict_file: str = None) -> None:
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        self.count_dict = {}
        if dict_file is not None:
            self.load_dict(dict_file)

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

    def load_json_dict(self, filename: str) -> None:
        """
        Load a count dictionary that was saved with ``save_dict`` or ``generate_count_dict``.

        JSON object keys are strings, so the fingerprint bit indices are
        converted back to ints to match the in-memory representation.

        :param filename: Name of the JSON file containing the count dictionary.
        """
        with open(filename) as f:
            raw_dict = json.load(f)
        self.count_dict = {int(k): int(v) for k, v in raw_dict.items()}



    def score(self, smiles_in):
        """Fraction of a molecule's fingerprint bits that are absent from the count dictionary.

        :param smiles_in: SMILES string to score
        :return: fraction of unseen bits, from 0.0 (all bits known) to 1.0. A SMILES
            that will not parse scores 1; a molecule with no fingerprint bits at all
            (an empty SMILES) scores 0, having nothing unusual in it.
        """
        mol = Chem.MolFromSmiles(smiles_in)
        if mol is None:
            return 1
        fp = self.fpgen.GetFingerprint(mol)
        on_bits = list(fp.GetOnBits())
        if not on_bits:
            # an empty molecule has no bits to be unusual
            return 0
        silly_bits = [x for x in on_bits if x not in self.count_dict]
        return len(silly_bits) / len(on_bits)

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


__all__ = ["SillyWalks"]


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
