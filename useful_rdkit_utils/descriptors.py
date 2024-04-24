import logging
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdchem import Mol
from tqdm.auto import tqdm


# ----------- Descriptors and fingerprints
def mol2morgan_fp(mol: Mol, radius: int = 2, nBits: int = 2048) -> DataStructs.ExplicitBitVect:
    """Convert an RDKit molecule to a Morgan fingerprint

    :param mol: RDKit molecule
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: RDKit Morgan fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return fp


def smi2morgan_fp(smi: str, radius: int = 2, nBits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """Convert a SMILES to a Morgan fingerprint

    :param smi: SMILES
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: RDKit Morgan fingerprint
    """
    mol = Chem.MolFromSmiles(smi)
    fp = None
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return fp


def mol2numpy_fp(mol: Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert an RDKit molecule to a numpy array with Morgan fingerprint bits
    Borrowed from https://iwatobipen.wordpress.com/2019/02/08/convert-fingerprint-to-numpy-array-and-conver-numpy-array-to-fingerprint-rdkit-memorandum/

    :param mol: RDKit molecule
    :param radius: fingerprint radius
    :param n_bits: number of fingerprint bits
    :return: numpy array with RDKit fingerprint bits
    """
    arr = np.zeros((0,), dtype=np.int8)
    fp = mol2morgan_fp(mol=mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smi2numpy_fp(smi: str, radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Convert a SMILES to a numpy array with Morgan fingerprint bits

    :param smi: SMILES string
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: numpy array with RDKit fingerprint bits
    """
    mol = Chem.MolFromSmiles(smi)
    arr = None
    if mol:
        arr = np.zeros((0,), dtype=np.int8)
        fp = mol2morgan_fp(mol=mol, radius=radius, nBits=nBits)
        DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# Code borrowed from Brian Kelley's Descriptastorus
# https://github.com/bp-kelley/descriptastorus
FUNCS = {name: func for name, func in Descriptors.descList}


def apply_func(name, mol):
    """Apply an RDKit descriptor calculation to a molecule

    :param name: descriptor name
    :param mol: RDKit molecule
    :return:
    """
    try:
        return FUNCS[name](mol)
    except:
        logging.exception("function application failed (%s->%s)", name, Chem.MolToSmiles(mol))
        return None


class RDKitDescriptors:
    """ Calculate RDKit descriptors"""

    def __init__(self):
        self.desc_names: List[str] = [desc_name for desc_name, _ in sorted(Descriptors.descList)]

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """Calculate descriptors for an RDKit molecule

        :param mol: RDKit molecule
        :return: a numpy array with descriptors
        """
        res = [apply_func(name, mol) for name in self.desc_names]
        return np.array(res, dtype=float)

    def calc_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """Calculate descriptors for a SMILES string

        :param smiles: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return self.calc_mol(mol)
        else:
            return None

    def pandas_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate descriptors for a list of SMILES strings and return them as a pandas DataFrame.

        :param smiles_list: List of SMILES strings
        :return: DataFrame with calculated descriptors. Each row corresponds to a SMILES string and each column to a descriptor.
        """
        desc_list = []
        for smi in tqdm(smiles_list):
            desc = self.calc_smiles(smi)
            if desc is not None:
                desc_list.append(desc)
            else:
                desc_list.append([None] * len(self.desc_names))
        df = pd.DataFrame(desc_list, columns=self.desc_names)
        return df


def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
    """
    Calculate descriptors for a list of RDKit molecules and return them as a pandas DataFrame.

    :param mol_list: List of RDKit molecules
    :return: DataFrame with calculated descriptors. Each row corresponds to a molecule and each column to a descriptor.
    """
    desc_list = []
    desc = None
    for mol in tqdm(mol_list):
        if mol:
            desc = self.calc_mol(mol)
        else:
            desc = [None] * len(self.desc_names)
        desc_list.append(desc)
    df = pd.DataFrame(desc_list, columns=self.desc_names)
    return df


class RDKitProperties:
    """ Calculate RDKit properties """

    def __init__(self):
        self.property_names: List[str] = list(rdMolDescriptors.Properties.GetAvailableProperties())
        self.property_getter: rdMolDescriptors.Properties = rdMolDescriptors.Properties(self.property_names)

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """Calculate properties for an RDKit molecule

        :param mol: RDKit molecule
        :return: a numpy array with properties
        """
        return np.array(self.property_getter.ComputeProperties(mol))

    def calc_smiles(self, smi: str) -> Optional[np.ndarray]:
        """Calculate properties for a SMILES string

        :param smi: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return np.array(self.calc_mol(mol))
        else:
            return None


class Ro5Calculator:
    """
    Calculate Rule of 5 properties + TPSA
    """

    def __init__(self):
        self.names: List[str] = ["MolWt", "LogP", "HBD", "HBA", "TPSA"]
        self.functions: List[Callable[[Mol], float]] = [MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA]

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """Calculate properties for a RDKit molecule

        :param mol: RDKit molecule
        :return: a numpy array with properties
        """
        return np.array([x(mol) for x in self.functions])

    def calc_smiles(self, smi: str) -> Optional[np.ndarray]:
        """Calculate properties for a SMILES string

        :param smi: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return self.calc_mol(mol)
        else:
            return None
