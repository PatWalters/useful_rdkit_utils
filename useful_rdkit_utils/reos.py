import sys
from pathlib import Path

import pandas as pd
import pystow
from rdkit import Chem


class REOS:
    """REOS - Rapid Elimination Of Swill\n
    Walters, Ajay, Murcko, "Recognizing molecules with druglike properties"\n
    Curr. Opin. Chem. Bio., 3 (1999), 384-387\n
    https://doi.org/10.1016/S1367-5931(99)80058-1
    """

    def __init__(self, active_rules=None):
        if active_rules is None:
            active_rules = ['Glaxo']
        url = 'https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv'
        self.rule_path = pystow.ensure('useful_rdkit_utils', 'data', url=url)
        self.rule_df = pd.read_csv(self.rule_path)
        self.read_rules(self.rule_path, active_rules)

    def parse_smarts(self):
        """Parse the SMARTS strings in the rules file to molecule objects and check for validity

        :return: True if all SMARTS are parsed, False otherwise
        """
        smarts_mol_list = []
        smarts_are_ok = True
        for idx, smarts in enumerate(self.rule_df.smarts, 1):
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                smarts_are_ok = False
                print(f"Error processing SMARTS on line {idx}", file=sys.stderr)
            smarts_mol_list.append(mol)
        self.rule_df['pat'] = smarts_mol_list
        return smarts_are_ok

    def read_rules(self, rules_file, active_rules=None):
        """Read a rules file

        :param rules_file: name of the rules file
        :param active_rules: list of active rule sets, all rule sets are used if
            this is None
        :return: None
        """
        if self.parse_smarts():
            self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules")
            if len(self.active_rule_df) == 0:
                available_rules = sorted(list(self.rule_df["rule_set_name"].unique()))
                raise ValueError(f"Supplied rules: {active_rules} not available. Please select from {available_rules}")

        else:
            print("Error reading rules, please fix the SMARTS errors reported above", file=sys.stderr)
            sys.exit(1)
        if active_rules is not None:
            self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules")
        else:
            self.active_rule_df = self.rule_df

    def set_active_rule_sets(self, active_rules=None):
        """Set the active rule set(s)

        :param active_rules: list of active rule sets
        :return: None
        """
        assert active_rules
        self.active_rule_df = self.rule_df.query("rule_set_name in @active_rules")

    def get_available_rule_sets(self):
        """Get the available rule sets in rule_df

        :return: a list of available rule sets
        """
        return self.rule_df.rule_set_name.unique()

    def get_active_rule_sets(self):
        """Get the active rule sets in active_rule_df

        :return: a list of active rule sets
        """
        return self.active_rule_df.rule_set_name.unique()

    def get_rule_file_location(self):
        """Get the path to the rules file as a Path

        :return: Path for rules file
        """
        return self.rule_path

    def process_mol(self, mol):
        """Match a molecule against the active rule set

        :param mol: input RDKit molecule
        :return: the first rule matched or "ok" if no rules are matched
        """
        cols = ['description', 'rule_set_name', 'pat', 'max']
        for desc, rule_set_name, pat, max_val in self.active_rule_df[cols].values:
            if len(mol.GetSubstructMatches(pat)) > max_val:
                return rule_set_name, desc
        return "ok", "ok"

    def process_smiles(self, smiles):
        """Convert SMILES to an RDKit molecule and call process_mol

        :param smiles: input SMILES
        :return: process_mol result or None if the SMILES can't be parsed
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error parsing SMILES {smiles}")
            return None
        return self.process_mol(mol)
