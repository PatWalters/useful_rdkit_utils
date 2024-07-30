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

    def __init__(self: "RDKitDescriptors", hide_progress: bool = False) -> None:
        """
        Initialize the RDKitDescriptors class.

        :param self: An instance of the RDKitDescriptors class
        :type self: RDKitDescriptors
        :param hide_progress: Flag to hide progress bar
        :type hide_progress: bool
        :return: None
        :rtype: None
        """
        self.hide_progress = hide_progress
        self.desc_names: List[str] = [desc_name for desc_name, _ in sorted(Descriptors.descList)]

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """Calculate descriptors for an RDKit molecule

        :param mol: RDKit molecule
        :return: a numpy array with descriptors
        """
        if mol is not None:
            res = np.array([apply_func(name, mol) for name in self.desc_names], dtype=float)
        else:
            res = np.array([None] * len(self.desc_names))
        return res

    def calc_smiles(self, smiles: str) -> np.ndarray:
        """Calculate descriptors for a SMILES string

        :param smiles: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smiles)
        return self.calc_mol(mol)

    def pandas_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculate descriptors for a list of SMILES strings and return them as a pandas DataFrame.

        :param smiles_list: List of SMILES strings
        :return: DataFrame with calculated descriptors. Each row corresponds to a SMILES string and each column to a descriptor.
        """
        desc_list = []
        for smi in tqdm(smiles_list, disable=self.hide_progress):
            desc_list.append(self.calc_smiles(smi))
        df = pd.DataFrame(desc_list, columns=self.desc_names)
        return df

    def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
        """
        Calculate descriptors for a list of RDKit molecules and return them as a pandas DataFrame.

        :param mol_list: List of RDKit molecules
        :return: DataFrame with calculated descriptors. Each row corresponds to a molecule and each column to a descriptor.
        """
        desc_list = []
        for mol in tqdm(mol_list, disable=self.hide_progress):
            desc_list.append(self.calc_mol(mol))
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
        if mol is not None:
            res = np.array(self.property_getter.ComputeProperties(mol))
        else:
            res = np.array([None] * len(self.property_names))
        return res

    def calc_smiles(self, smi: str) -> Optional[np.ndarray]:
        """Calculate properties for a SMILES string

        :param smi: SMILES string
        :return: a numpy array with properties
        """
        mol = Chem.MolFromSmiles(smi)
        return self.calc_mol(mol)

    def pandas_smiles(self, smi_list: List[str]) -> pd.DataFrame:
        """
        Calculates properties for a list of SMILES strings and returns them as a pandas DataFrame.

        :param smi_list: List of SMILES strings
        :type smi_list: List[str]
        :return: DataFrame with calculated properties. Each row corresponds to a SMILES string and each column to a property.
        :rtype: pd.DataFrame
        """
        prop_list = []
        for smi in tqdm(smi_list):
            prop_list.append(self.calc_smiles(smi))
        return pd.DataFrame(prop_list, columns=[self.property_names])

    def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
        """
        Calculates properties for a list of RDKit molecules and returns them as a pandas DataFrame.

        :param mol_list: List of RDKit molecules
        :type mol_list: List[Mol]
        :return: DataFrame with calculated properties. Each row corresponds to a molecule and each column to a property.
        :rtype: pd.DataFrame
        """
        prop_list = []
        for mol in tqdm(mol_list):
            prop_list.append(self.calc_mol(mol))
        return pd.DataFrame(prop_list, columns=self.property_names)


class Ro5Calculator:
    """
    A class used to calculate Lipinski's Rule of Five properties for a given molecule.

    Attributes
    ----------
    names : List[str]
        A list of names of the properties to be calculated.
    functions : List[Callable[[Mol], float]]
        A list of functions used to calculate the properties.

    Methods
    -------
    calc_mol(mol: Mol) -> np.ndarray
        Calculates properties for a RDKit molecule.

    calc_smiles(smi: str) -> Optional[np.ndarray]
        Calculates properties for a SMILES string.

    pandas_smiles(smiles_list: List[str]) -> pd.DataFrame
        Calculates properties for a list of SMILES strings and returns them as a pandas DataFrame.

    pandas_mols(mol_list: List[Mol]) -> pd.DataFrame
        Calculates properties for a list of RDKit molecules and returns them as a pandas DataFrame.
    """

    def __init__(self: "Ro5Calculator") -> None:
        """
        Initialize the Ro5Calculator class.

        :param self: An instance of the Ro5Calculator class
        :type self: Ro5Calculator
        :return: None
        :rtype: None
        """
        self.names: List[str] = ["MolWt", "LogP", "HBD", "HBA", "TPSA"]
        self.functions: List[Callable[[Mol], float]] = [MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA]

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """
        Calculate properties for a RDKit molecule

        :param mol: RDKit molecule
        :type mol: Mol
        :return: a numpy array with properties
        :rtype: np.ndarray
        """
        if mol is not None:
            res = np.array([x(mol) for x in self.functions])
        else:
            res = np.array([None] * len(self.names))
        return res

    def calc_smiles(self, smi: str) -> Optional[np.ndarray]:
        """
        Calculate properties for a SMILES string

        :param smi: SMILES string
        :type smi: str
        :return: a numpy array with properties
        :rtype: Optional[np.ndarray]
        """
        mol = Chem.MolFromSmiles(smi)
        return self.calc_mol(mol)

    def pandas_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Calculates properties for a list of SMILES strings and returns them as a pandas DataFrame.

        :param smiles_list: List of SMILES strings
        :type smiles_list: List[str]
        :return: DataFrame with calculated properties. Each row corresponds to a SMILES string and each column to a property.
        :rtype: pd.DataFrame
        """
        prop_list = []
        for smiles in tqdm(smiles_list):
            prop_list.append(self.calc_smiles(smiles))
        return pd.DataFrame(prop_list, columns=self.names)

    def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
        """
        Calculates properties for a list of RDKit molecules and returns them as a pandas DataFrame.

        :param mol_list: List of RDKit molecules
        :type mol_list: List[Mol]
        :return: DataFrame with calculated properties. Each row corresponds to a molecule and each column to a property.
        :rtype: pd.DataFrame
        """
        prop_list = []
        for mol in tqdm(mol_list):
            prop_list.append(self.calc_mol(mol))
        return pd.DataFrame(prop_list, columns=self.names)
