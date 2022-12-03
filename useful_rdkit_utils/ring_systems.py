#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
import pandas as pd
from tqdm.auto import tqdm
import pystow
import sys
import click
from operator import itemgetter

import useful_rdkit_utils


class RingSystemFinder:
    """A class to identify ring systems """

    def __init__(self):
        """Initialize susbstructure search objects to identify key functionality
        """
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#18R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        """Assign the property "protected" to all ring carbonyls, etc.
        :param mol: input molecule
        :return: None
        """
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        """Cleave bonds that are not in rings and not protected
        :param mol: input molecule
        :return: None
        """
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())

        if len(frag_bond_list):
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            # Chem.AssignStereochemistry(frag_mol, cleanIt=True, force=True)
            return frag_mol
        else:
            return mol

    def cleanup_fragments(self, mol, keep_dummy=False):
        """Split a molecule containing multiple ring systems into individual ring systems
        :param keep_dummy: retain dummy atoms
        :param mol: input molecule
        :return: a list of SMILES corresponding to individual ring systems
        """
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        if keep_dummy:
                            atm.SetProp("atomLabel", "R")
                        else:
                            atm.SetAtomicNum(1)
                        atm.SetIsotope(0)
                # Convert explict Hs to implicit
                frag = Chem.RemoveAllHs(frag)
                frag = self.fix_bond_stereo(frag)
                ring_system_list.append(frag)
        return ring_system_list

    @staticmethod
    def fix_bond_stereo(mol):
        """Loop over double bonds and change stereo specification for double bonds that don't have stereo
        :param mol: input RDKit molecule
        :return: output RDKit molecule
        """
        for bnd in mol.GetBonds():
            if bnd.GetBondType() == Chem.BondType.DOUBLE:
                begin_atm = bnd.GetBeginAtom()
                end_atm = bnd.GetEndAtom()
                # Look for double bond atoms with two attached hydrogens
                if begin_atm.GetDegree() == 1 or end_atm.GetDegree() == 1:
                    bnd.SetStereo(Chem.BondStereo.STEREONONE)
        return mol

    def find_ring_systems(self, mol, keep_dummy=False, as_mols=False):
        """Find the ring systems for an RDKit molecule
        :param as_mols: return results a molecules (otherwise return SMILES)
        :param keep_dummy: retain dummy atoms
        :param mol: input molecule
        :return: a list of SMILES corresponding to individual ring systems
        """
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        output_list = self.cleanup_fragments(frag_mol, keep_dummy=keep_dummy)
        if not as_mols:
            output_list = [Chem.MolToSmiles(x) for x in output_list]
        return output_list


def test_ring_system_finder():
    """A quick test for the RingSystemFinder class
    :return: None
    """
    mol = Chem.MolFromSmiles("CC(=O)[O-].CCn1c(=O)/c(=C2\Sc3ccccc3N2C)s/c1=C\C1CCC[n+]2c1sc1ccccc12")
    ring_system_finder = RingSystemFinder()
    ring_system_finder.find_ring_systems(mol)


def create_ring_dictionary(input_smiles, output_csv):
    """Read a SMILES file, extract ring systems, write out ring systems and frequency
    :param input_smiles: input SMILES file
    :param output_csv: output csv file
    :return: None
    """
    ring_system_finder = RingSystemFinder()
    df = pd.read_csv(input_smiles, sep=" ", names=["SMILES", "Name"])
    ring_system_output_list = []
    inchi_smi_dict = {}
    for smi in tqdm(df.SMILES):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ring_system_mols = ring_system_finder.find_ring_systems(mol, as_mols=True)
        ring_system_inchi_list = [Chem.MolToInchiKey(x) for x in ring_system_mols]
        ring_system_smiles_list = [Chem.MolToSmiles(x) for x in ring_system_mols]
        for inchi_val, smi_val in zip(ring_system_inchi_list, ring_system_smiles_list):
            inchi_smi_dict[inchi_val] = smi_val
        ring_system_output_list += ring_system_inchi_list
    df_out = pd.DataFrame(pd.Series(ring_system_output_list).value_counts())
    df_out.index.name = "InChI"
    df_out.columns = ['Count']
    df_out = df_out.reset_index()
    df_out['SMILES'] = [inchi_smi_dict.get(x) for x in df_out.InChI]
    df_out[['SMILES', 'InChI', 'Count']].to_csv(output_csv, index=False)


class RingSystemLookup:
    """Lookup ring systems from a dictionary of rings and frequencies"""

    def __init__(self, ring_system_csv=None):
        """
        Initialize the lookup table
        :param ring_system_csv: csv file with ring smiles and frequency
        """
        if ring_system_csv is None:
            url = 'https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/master/data/chembl_ring_systems.csv'
            self.rule_path = pystow.ensure('useful_rdkit_utils', 'data', url=url)
        else:
            self.rule_path = ring_system_csv
        ring_df = pd.read_csv(self.rule_path)
        self.ring_dict = dict(ring_df[["InChI", "Count"]].values)
        self.ring_system_finder = RingSystemFinder()

    def process_mol(self, mol):
        """
        find ring systems in an RDKit molecule
        :param mol: input molecule
        :return: list of SMILES for ring systems
        """
        output_ring_list = []
        if mol:
            ring_system_list = self.ring_system_finder.find_ring_systems(mol, as_mols=True)
            for ring in ring_system_list:
                smiles = Chem.MolToSmiles(ring)
                inchi = Chem.MolToInchiKey(ring)
                count = self.ring_dict.get(inchi) or 0
                output_ring_list.append((smiles, count))
        return output_ring_list

    def process_smiles(self, smi):
        """
        find ring systems from a SMILES
        :param smi: input SMILES
        :return: list of SMILES for ring systems
        """
        res = []
        mol = Chem.MolFromSmiles(smi)
        if mol:
            res = self.process_mol(mol)
        return res


def test_ring_system_lookup(input_filename, output_filename):
    """Test for RingSystemLookup
    :param input_filename: input smiles file
    :param output_filename: output csv file
    :return: None
    """
    df = pd.read_csv(input_filename, sep=" ", names=["SMILES", "Name"])
    ring_system_lookup = RingSystemLookup()
    min_freq_list = []
    for smi in tqdm(df.SMILES):
        freq_list = ring_system_lookup.process_smiles(smi)
        min_freq_list.append(get_min_ring_frequency(freq_list))
    df['min_ring'] = [x[0] for x in min_freq_list]
    df['min_freq'] = [x[1] for x in min_freq_list]
    df.to_csv(output_filename, index=False)


def get_min_ring_frequency(ring_list):
    """Get minimum frequency from RingSystemLookup.process_smiles
    :param ring_list: output from RingSystemLookup.process_smiles
    :return: [ring_with minimum frequency, minimum frequency], acyclic molecules return ["",-1]
    """
    ring_list.sort(key=itemgetter(1))
    if len(ring_list):
        return ring_list[0]
    else:
        return ["", -1]


@click.command()
@click.option("--mode", prompt="mode [build|search]", help="[build|search]")
@click.option("--infile", prompt="Input csv file", help="input file")
@click.option("--outfile", prompt="Output csv file", help="output file")
def main(mode, infile, outfile):
    mode_list = ["build", "search"]
    if mode not in mode_list:
        print(f"mode must be one of {mode_list}")
        sys.exit(0)
    if mode == "build":
        create_ring_dictionary(infile, outfile)
    if mode == "search":
        test_ring_system_lookup(infile, outfile)


if __name__ == "__main__":
    main()
