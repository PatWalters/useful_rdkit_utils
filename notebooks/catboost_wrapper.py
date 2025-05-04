#!/usr/bin/env python

from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class CatBoostWrapper:
    def __init__(self, y_col):
        self.cb = CatBoostRegressor(verbose=False)
        self.y_col = y_col
        self.fp_name = "fp"
        self.fg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def fit(self, train):
        train['mol'] = train.SMILES.apply(Chem.MolFromSmiles)
        train[self.fp_name] = train.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        self.cb.fit(np.stack(train.fp),train[self.y_col])

    def predict(self, test):
        test['mol'] = test.SMILES.apply(Chem.MolFromSmiles)
        test[self.fp_name] = test.mol.apply(self.fg.GetCountFingerprintAsNumPy)
        pred = self.cb.predict(np.stack(np.stack(test[self.fp_name])))
        return pred

    def validate(self, train, test):
        self.fit(train)
        pred = self.predict(test)
        return pred



def main():
    df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/datafiles/refs/heads/main/biogen_logS.csv")
    train, test = train_test_split(df)
    cb_wrapper = CatBoostWrapper("logS")
    pred = cb_wrapper.validate(train, test)
    print(pred)

if __name__ == "__main__":
    main()
