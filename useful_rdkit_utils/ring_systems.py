#!/usr/bin/env python
import itertools
from operator import itemgetter

import pandas as pd
import pystow
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from tqdm.auto import tqdm


class RingSystemFinder:
    """A class to identify ring systems """

    def __init__(self):
        """Initialize susbstructure search objects to identify key functionality
        """
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#16R]=[OR0,SR0,CR0,NR0]")
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
    mol = Chem.MolFromSmiles(r"CC(=O)[O-].CCn1c(=O)/c(=C2\Sc3ccccc3N2C)s/c1=C\C1CCC[n+]2c1sc1ccccc12")
    ring_system_finder = RingSystemFinder()
    ring_system_finder.find_ring_systems(mol)


def create_ring_dictionary(input_chemreps, output_csv):
    """Read the ChEMBL chemreps.txt file, extract ring systems, write out ring systems and frequency
    :param input_chemreps: ChEMBL chemreps file
    :param output_csv: output csv file
    :return: None
    """
    ring_system_finder = RingSystemFinder()
    df = pd.read_csv(input_chemreps, sep="\t")
    ring_system_output_list = []
    inchi_smi_dict = {}
    for smi in tqdm(df.canonical_smiles):
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

    df_no_stereo_out = generate_no_stereo_ring_systems(df_out)
    df_no_stereo_out.to_csv(output_csv.replace(".csv", "_no_stereo.csv"), index=False)


class RingSystemLookup:
    """Lookup ring systems from a dictionary of rings and frequencies"""

    def __init__(self, ring_file=None, ignore_stereo=False):
        ring_csv_name = "chembl_ring_systems.csv"
        if ignore_stereo:
            ring_csv_name = ring_csv_name.replace(".csv", "_no_stereo.csv")
        if ring_file is None:
            url = f'https://raw.githubusercontent.com/PatWalters/useful_rdkit_utils/refs/heads/master/data/{ring_csv_name}'
            self.rule_path = pystow.ensure('useful_rdkit_utils', 'data', url=url)
        else:
            self.rule_path = ring_file
        self.ignore_stereo = ignore_stereo
        self.ring_df = pd.read_csv(self.rule_path)
        self.ring_dict = dict(self.ring_df[["InChI", "Count"]].values)
        self.ring_system_finder = RingSystemFinder()
        self.enumerator = rdMolStandardize.TautomerEnumerator()

    def process_mol(self, mol_in):
        """
        find ring systems in an RDKit molecule
        :param mol_in: input molecule
        :return: list of SMILES for ring systems
        """
        #mol = self.enumerator.Canonicalize(mol_in)
        mol = mol_in
        # mol = mol_in
        output_ring_list = []
        if mol:
            if self.ignore_stereo:
                Chem.RemoveStereochemistry(mol)
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

    def pandas_smiles_list(self, smiles_list):
        """
        find ring systems from a list of SMILES
        :param smiles_list: list of SMILES
        :return: dataframe with ring information
        """
        res = []
        for smi in tqdm(smiles_list):
            res.append(self.process_smiles(smi))
        res_df = pd.DataFrame()
        res_df["ring_systems"] = res
        freq_data = res_df.ring_systems.apply(get_min_ring_frequency)
        res_df["min_ring"] = [x[0] for x in freq_data]
        res_df["min_freq"] = [x[1] for x in freq_data]
        return res_df


def test_ring_system_lookup(input_filename, output_filename):
    """Test for RingSystemLookup
    :param input_filename: input smiles file
    :param output_filename: output csv file
    :return: None
    """
    df = pd.read_csv(input_filename, sep=" ", names=["SMILES", "Name"])
    ring_system_lookup = RingSystemLookup.default()
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


def remove_stereo_from_smiles(smi_in):
    mol = Chem.MolFromSmiles(smi_in)
    smi_out = None
    inchi_key = None
    if mol:
        Chem.RemoveStereochemistry(mol)
        smi_out = Chem.MolToSmiles(mol)
        inchi_key = Chem.MolToInchiKey(mol)
    return smi_out, inchi_key


def generate_no_stereo_ring_systems(df):
    tqdm.pandas()
    df[["no_stereo_smiles", "no_stereo_inchi"]] = df.SMILES.progress_apply(remove_stereo_from_smiles).to_list()
    res = []
    for k, v in tqdm(df.groupby("no_stereo_inchi")):
        res.append([v.no_stereo_smiles.values[0], k, v.Count.sum()])
    no_stereo_ring_df = pd.DataFrame(res, columns=["SMILES", "InChI", "Count"])
    no_stereo_ring_df.sort_values("Count", ascending=False, inplace=True)
    df.drop(["no_stereo_smiles", "no_stereo_inchi"], axis=1, inplace=True)
    return no_stereo_ring_df





# ----------- Ring stats
def get_spiro_atoms(mol):
    """Get atoms that are part of a spiro fusion

    :param mol: input RDKit molecule
    :return: a list of atom numbers for atoms that are the centers of spiro fusions
    """
    info = mol.GetRingInfo()
    ring_sets = [set(x) for x in info.AtomRings()]
    spiro_atoms = []
    for i, j in itertools.combinations(ring_sets, 2):
        i_and_j = (i.intersection(j))
        if len(i_and_j) == 1:
            spiro_atoms += list(i_and_j)
    return spiro_atoms


def max_ring_size(mol):
    """Get the size of the largest ring in a molecule

    :param mol: input_molecule
    :return: size of the largest ring or 0 for an acyclic molecule
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) == 0:
        return 0
    else:
        return max([len(x) for x in ri.AtomRings()])


def ring_stats(mol):
    """Get some simple statistics for rings

    :param mol: RDKit molecule
    :return: number of rings, maximum ring size
    """
    max_size = max_ring_size(mol)
    num_rings = CalcNumRings(mol)
    return num_rings, max_size



