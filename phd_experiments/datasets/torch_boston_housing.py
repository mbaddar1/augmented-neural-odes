import pandas as pd
import torch.nn
from torch.utils.data import Dataset


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
        self.X = torch.tensor(df.loc[:, X_cols].values,dtype=torch.float32)
        self.y = torch.tensor(df.loc[:, y_col].values,dtype=torch.float32)

    def get_Xdim(self):
        return self.X_dim

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == '__main__':
    ds = TorchBostonHousingPrices()
