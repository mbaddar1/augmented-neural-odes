import pandas as pd
import torch.nn
from torch.utils.data import Dataset
from sklearn import datasets


# https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data
# https://www.kaggle.com/code/marcinrutecki/regression-models-evaluation-metrics
class TorchBostonHousingPrices(Dataset):
    def __init__(self):
        df = pd.read_csv('boston.csv', sep=',')
        self.N = df.shape[0]
        X_cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
        y_col = ["MEDV"]
        self.X = torch.tensor(df.loc[:, X_cols].values)
        self.y = torch.tensor(df.loc[:, y_col].values)
        x=10

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__=='__main__':
    ds = TorchBostonHousingPrices()

if __name__ == '__main__':
    pass
