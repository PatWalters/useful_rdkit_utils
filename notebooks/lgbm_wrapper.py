#!/usr/bin/env python

from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import useful_rdkit_utils as uru

class LGBMPropWrapper:
    def __init__(self, y_col):
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.rdkit_desc = uru.RDKitDescriptors(hide_progress=True)
        self.desc_name = 'desc'

    def fit(self, train):
        train[self.desc_name] = self.rdkit_desc.pandas_smiles(train.SMILES).values.tolist()
        self.lgbm.fit(np.stack(train[self.desc_name]),train[self.y_col])

    def predict(self, test):
        test[self.desc_name] = self.rdkit_desc.pandas_smiles(test.SMILES).values.tolist()
        pred = self.lgbm.predict(np.stack(np.stack(test[self.desc_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred


class LGBMMorganCountWrapper:
    def __init__(self, y_col):
        self.lgbm = LGBMRegressor(verbose=-1)
        self.y_col = y_col
        self.fp_name = "fp"
        self.fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def fit(self, train):
        train['mol'] = train.SMILES.apply(Chem.MolFromSmiles)
        train[self.fp_name] = train.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        self.lgbm.fit(np.stack(train.fp),train[self.y_col])

    def predict(self, test):
        test['mol'] = test.SMILES.apply(Chem.MolFromSmiles)
        test[self.fp_name] = test.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        pred = self.lgbm.predict(np.stack(np.stack(test[self.fp_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred



def main():
    df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    train, test = train_test_split(df)
    lgbm_wrapper = LGBMMorganCountWrapper("logS")
    pred = lgbm_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
