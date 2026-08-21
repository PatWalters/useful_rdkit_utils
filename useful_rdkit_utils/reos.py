import sys
from typing import List, Optional

import pandas as pd
import pystow
from rdkit import Chem
from tqdm.auto import tqdm
from rdkit.Chem.rdchem import Mol



class REOS:
    """REOS - Rapid Elimination Of Swill\n
    Walters, Ajay, Murcko, "Recognizing molecules with druglike properties"\n
    Curr. Opin. Chem. Bio., 3 (1999), 384-387\n
    https://doi.org/10.1016/S1367-5931(99)80058-1
    """

    def __init__(self, active_rules: Optional[List[str]] = None) -> None:
        """
        Initialize the REOS class.

        :param active_rules: List of active rules. If None, the default rule 'Glaxo' is used.
        :type active_rules: Optional[List[str]]
        :default active_rules: None
        """
        self.output_smarts = False
        if active_rules is None:
            active_rules = ['Glaxo']
        url = 'https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv'
        self.rule_path = pystow.ensure('useful_rdkit_utils', 'data', url=url)
        self.active_rule_df = None
        self.active_rules = None
        self.rule_df = None
        self.read_rules(self.rule_path, active_rules)
        self.rules_dict = self.get_rules_dict()

    def set_output_smarts(self, output_smarts):
        """Determine whether SMARTS are returned
        :param output_smarts: True or False
        :return: None
        """
        self.output_smarts = output_smarts

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

        :param rules_file: name of the rules file to read
        :param active_rules: list of active rule sets, all rule sets are used if
            this is None
        :raises ValueError: if the file contains SMARTS that cannot be parsed, or if
            active_rules names a rule set that is not in the file
        :return: None
        """
        self.rule_path = rules_file
        self.rule_df = pd.read_csv(rules_file)
        if not self.parse_smarts():
            raise ValueError(
                f"Could not parse every SMARTS in {rules_file}; see the errors reported above"
            )
        if active_rules is not None:
            self._select_rule_sets(active_rules)
        else:
            # no specific rule set requested: use all rules
            self.active_rule_df = self.rule_df.copy()
            self.active_rules = sorted(self.rule_df["rule_set_name"].unique())

    def _select_rule_sets(self, active_rules) -> None:
        """Point active_rule_df at the named rule sets, validating the names.

        :param active_rules: rule set name, or list of rule set names
        :raises ValueError: if none of the names match a rule set in the rules file
        :return: None
        """
        # a bare rule set name is a natural thing to pass, and list("PAINS") would
        # otherwise store the individual characters as the active rule sets
        if isinstance(active_rules, str):
            active_rules = [active_rules]
        active_rule_df = self.rule_df.query("rule_set_name in @active_rules").copy()
        if len(active_rule_df) == 0:
            available_rules = sorted(list(self.rule_df["rule_set_name"].unique()))
            raise ValueError(f"Supplied rules: {active_rules} not available. Please select from {available_rules}")
        self.active_rule_df = active_rule_df
        self.active_rules = list(active_rules)

    def get_rules_dict(self) -> dict:
        """Create a dictionary from the rules in rule_df.

        :return: Dictionary with (rule_set_name, description) tuples as keys and smarts as values
        """
        rules_dict = {}
        for _, row in self.rule_df.iterrows():
            key = (row['rule_set_name'], row['description'])
            rules_dict[key] = row['smarts']
        return rules_dict

    def set_active_rule_sets(self, active_rules=None):
        """Set the active rule set(s)

        :param active_rules: rule set name, or list of rule set names
        :return: None
        """
        assert active_rules
        self._select_rule_sets(active_rules)

    def set_min_priority(self, min_priority: int) -> None:
        """Set the minimum priority for rules to be included in the active rule set.

        :param min_priority: The minimum priority for rules to be included.
        :return: None
        """
        # reset active_rule_df to the rule sets that were selected most recently
        self._select_rule_sets(self.active_rules)
        # filter to only include rules with priority greater than or equal to min_priority
        self.active_rule_df = self.active_rule_df.query("priority >= @min_priority").copy()

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

    def drop_rule(self, description: str) -> None:
        """Drops a rule from the active rule set based on its description.

        :param: description: The description of the rule to be dropped.
        :return: None
        """
        num_rules_before = len(self.active_rule_df)
        self.active_rule_df = self.active_rule_df.query("description != @description")
        num_rules_after = len(self.active_rule_df)
        print(f"Dropped {num_rules_before - num_rules_after} rule(s)")

    def get_rule_file_location(self):
        """Get the path to the rules file as a Path

        :return: Path for rules file
        """
        return self.rule_path

    def process_mol(self, mol):
        """Match a molecule against the active rule set

        :param mol: input RDKit molecule
        :return: the first rule matched, "ok" if no rules are matched, or None if
            mol is None (e.g. a SMILES that failed to parse)
        """
        if mol is None:
            return None
        cols = ['description', 'rule_set_name', 'smarts', 'pat', 'max']
        if self.output_smarts:
            ret_val = ("ok", "ok", "ok")
        else:
            ret_val = ("ok", "ok")
        for desc, rule_set_name, smarts, pat, max_val in self.active_rule_df[cols].values:
            if len(mol.GetSubstructMatches(pat)) > max_val:
                if self.output_smarts:
                    ret_val = rule_set_name, desc, smarts
                else:
                    ret_val = rule_set_name, desc
                break
        return ret_val

    def process_smiles(self, smiles):
        """Convert SMILES to an RDKit molecule and call process_mol

        :param smiles: input SMILES
        :return: process_mol result or None if the SMILES can't be parsed
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error parsing SMILES {smiles}", file=sys.stderr)
            return None
        return self.process_mol(mol)

    def pandas_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """Process a list of SMILES strings

        :param smiles_list: list of SMILES strings
        :return: a pandas DataFrame with the results
        """
        results = []
        for smiles in tqdm(smiles_list):
            results.append(self.process_smiles(smiles))
        if self.output_smarts:
            column_names = ['rule_set_name', 'description', 'smarts']
        else:
            column_names = ['rule_set_name', 'description']
        return pd.DataFrame(results, columns=column_names)

    def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
        """Process a list of RDKit molecules

        :param mol_list: list of RDKit molecules
        :return: a pandas DataFrame with the results
        """
        results = []
        for mol in tqdm(mol_list):
            results.append(self.process_mol(mol))
        if self.output_smarts:
            column_names = ['rule_set_name', 'description', 'smarts']
        else:
            column_names = ['rule_set_name', 'description']
        return pd.DataFrame(results, columns=column_names)


__all__ = ["REOS"]
