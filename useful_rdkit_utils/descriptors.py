from typing import List, Optional, Callable, Any

import numpy as np
import pandas as pd
from pandas import Series
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdchem import Mol
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler


# ----------- Descriptors and fingerprints


class Smi2Fp:
    """ Calculate Morgan fingerprints from SMILES strings """

    def __init__(self, radius: int = 3, fpSize: int = 2048):
        self.fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)

    def get_np(self, smiles):
        """
        Convert a SMILES string to a numpy array with Morgan fingerprint bits.

        :param smiles: SMILES string
        :return: numpy array with Morgan fingerprint bits
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol:
            fp = self.fpgen.GetFingerprintAsNumPy(mol)
        return fp

    def get_np_counts(self, smiles):
        """
        Convert a SMILES string to a numpy array with Morgan fingerprint counts.

        :param smiles: SMILES string
        :return: numpy array with Morgan fingerprint counts
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol:
            fp = self.fpgen.GetCountFingerprintAsNumPy(mol)
        return fp

    def get_fp(self, smiles):
        """
        Convert a SMILES string to a Morgan fingerprint.

        :param smiles: SMILES string
        :return: Morgan fingerprint
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol:
            fp = self.fpgen.GetFingerprint(mol)
        return fp

    def get_count_fp(self, smiles):
        """
        Convert a SMILES string to a Morgan count fingerprint.

        :param smiles: SMILES string
        :return: Morgan count fingerprint
        """
        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol:
            fp = self.fpgen.GetCountFingerprint(mol)
        return fp


def mol2morgan_fp(mol: Mol, radius: int = 2, nBits: int = 2048) -> DataStructs.ExplicitBitVect:
    """
    Convert an RDKit molecule to a Morgan fingerprint
    To avoid the rdkit deprecated warning, do this
    from rdkit import rdBase
    with rdBase.BlockLogs():
        uru.smi2numpy_fp("CCC")

    :param mol: RDKit molecule
    :param radius: fingerprint radius
    :param nBits: number of fingerprint bits
    :return: RDKit Morgan fingerprint
    """
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol)
    return fp


def smi2morgan_fp(smi: str, radius: int = 2, nBits: int = 2048) -> Optional[DataStructs.ExplicitBitVect]:
    """Convert a SMILES to a Morgan fingerprint
    To avoid the rdkit deprecated warning, do this
    from rdkit import rdBase
    with rdBase.BlockLogs():
        uru.smi2numpy_fp("CCC")

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


class RDKitDescriptors:
    """Calculate RDKit descriptors for molecules or SMILES.

    Provide methods to compute descriptor vectors for a single molecule or SMILES,
    and to produce pandas DataFrames for lists of molecules or SMILES.

    Attributes
    ----------
    desc_names
        Sorted list of descriptor names that will be calculated.
    hide_progress
        Whether to hide progress bars when processing lists.
    """

    def __init__(self: "RDKitDescriptors",
                 desc_names=None,
                 hide_progress: bool = False,
                 skip_fragments=False) -> None:
        """
        Initialize descriptor calculator.

        :param desc_names: Optional list of descriptor names to use. If not provided, the full RDKit descriptor list is used.
        :param hide_progress: If true, progress bars are disabled when processing lists.
        :param skip_fragments: If true, descriptors whose names contain "fr_" are excluded.
        :return: None
        """
        self.hide_progress = hide_progress
        if desc_names is not None:
            self.desc_names = desc_names
        else:
            self.desc_names: List[str] = sorted([x[0] for x in Descriptors.descList])
        if skip_fragments:
            self.desc_names = [x for x in self.desc_names if "fr_" not in x]

    def update_descriptors(self, index_list: List[int]) -> None:
        """Update the descriptor names to only include those at the specified indices.

        :param index_list: List of indices to keep
        :return: None
        """
        self.desc_names = [self.desc_names[i] for i in index_list]

    def calc_mol(self, mol: Mol) -> np.ndarray:
        """Calculate descriptors for an RDKit molecule.

        :param mol: RDKit molecule
        :return: A numpy array with descriptor values
        """
        if mol is not None:
            RDLogger.DisableLog('rdApp.warning')
            res_dict = Descriptors.CalcMolDescriptors(mol)
            RDLogger.EnableLog('rdApp.warning')
            res = np.array([res_dict[x] for x in self.desc_names])
        else:
            res = np.array([None] * len(self.desc_names))
        return res

    def calc_smiles(self, smiles: str) -> np.ndarray:
        """Calculate descriptors for a SMILES string.

        :param smiles: SMILES string
        :return: A numpy array with descriptor values
        """
        mol = Chem.MolFromSmiles(smiles)
        return self.calc_mol(mol)

    def pandas_smiles(self, smiles_list: List[str]) -> pd.DataFrame:
        """Calculate descriptors for a list of SMILES and return a DataFrame.

        :param smiles_list: List of SMILES strings
        :return: DataFrame where each row corresponds to a SMILES and columns are descriptors
        """
        desc_list = []
        for smi in tqdm(smiles_list, disable=self.hide_progress):
            desc_list.append(self.calc_smiles(smi))
        df = pd.DataFrame(desc_list, columns=self.desc_names)
        return df

    def pandas_mols(self, mol_list: List[Mol]) -> pd.DataFrame:
        """Calculate descriptors for a list of RDKit molecules and return a DataFrame.

        :param mol_list: List of RDKit molecule objects
        :return: DataFrame where each row corresponds to a molecule and columns are descriptors
        """
        desc_list = []
        for mol in tqdm(mol_list, disable=self.hide_progress):
            desc_list.append(self.calc_mol(mol))
        df = pd.DataFrame(desc_list, columns=self.desc_names)
        return df


def clean_descriptors(desc_in: np.ndarray) -> tuple[np.ndarray | List[int]]:
    """
    Remove descriptor columns that contain any NaN or infinite values.

    :param desc_in: Input descriptor array
    :return: Tuple containing the cleaned descriptor array (only columns with all finite values) and a list of kept column indices
    """
    valid_mask = ~np.any(np.isnan(desc_in) | np.isinf(desc_in), axis=0)
    if not np.any(valid_mask):
        raise ValueError("All descriptor columns contain NaN or infinite values.")
    clean_desc = desc_in[:, valid_mask]
    kept_indices = np.where(valid_mask)[0].tolist()
    return clean_desc, kept_indices


def scale_descriptors(desc_in: pd.DataFrame) -> tuple[Any, StandardScaler]:
    """Scale descriptor DataFrame using StandardScaler.

    :param desc_in: Input descriptor DataFrame
    :return: Tuple with the scaled descriptor array and the fitted StandardScaler
    """
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_in)
    return desc_scaled, scaler


def clean_and_scale_descriptors(desc_in: pd.DataFrame) -> tuple[Any, StandardScaler]:
    """
    Clean and scale a descriptor DataFrame.

    :param desc_in: Input descriptor DataFrame
    :return: Tuple containing the cleaned and scaled descriptor array and the fitted StandardScaler
    """
    desc_clean, _ = clean_descriptors(desc_in)
    desc_scaled, scaler = scale_descriptors(desc_clean)
    return desc_scaled, scaler


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


def compare_datasets(
        train_fp: list,
        test_fp: list
) -> list:
    """
    Compare two datasets of fingerprints and return the maximum Tanimoto similarity for each test fingerprint.

    :param train_fp: Training set fingerprints.
    :param test_fp: Test set fingerprints.
    :return: List of maximum similarity values for each test fingerprint.
    """
    nbr_sim_list = []
    if isinstance(train_fp, pd.Series):
        train_fp = train_fp.values
    for fp in test_fp:
        sim_list = DataStructs.BulkTanimotoSimilarity(fp, train_fp)
        nbr_sim_list.append(max(sim_list))
    return nbr_sim_list
