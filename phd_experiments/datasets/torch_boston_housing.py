import pandas as pd
import torch.nn
from sklearn import preprocessing
from torch.utils.data import Dataset
import numpy as np

# https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data
# https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-
# NN for boston housing
# https://github.com/glingden/Boston-House-Price-Prediction
class TorchBostonHousingPrices(Dataset):
    def __init__(self, csv_file: str):
        df = pd.read_csv(csv_file, sep=',')
        self.N = df.shape[0]
        X_cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
        y_col = ["MEDV"]
        self.X_dim = len(X_cols)
        X = df.loc[:, X_cols].values
        scaler = preprocessing.StandardScaler().fit(X)
        X_scale = scaler.transform(X)
        self.X = torch.tensor(X_scale,dtype=torch.float32)
        y = df.loc[:, y_col].values
        scaler = preprocessing.StandardScaler().fit(y)
        y_scale = scaler.transform(y)
        self.y = torch.tensor(y_scale,dtype=torch.float32)
        x=10
    def get_Xdim(self):
        return self.X_dim

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    ds = TorchBostonHousingPrices()
