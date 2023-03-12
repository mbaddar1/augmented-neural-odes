import torch
from sklearn.datasets import load_diabetes
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader


class TorchDiabetesDataset(Dataset):
    def __init__(self, dtype=torch.float32):
        X, y = load_diabetes(return_X_y=True)
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype).view(len(y),1)

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# quick test

if __name__ == '__main__':
    ds = TorchDiabetesDataset()
    dl = DataLoader(dataset=ds, batch_size=32, shuffle=True)
    epochs = 100
    Dx = 10
    Dy = 1
    A = torch.distributions.Uniform(0.01, 0.05).sample(sample_shape=torch.Size([Dx + 1, Dy]))
    alpha = 0.8
    mse_loss_fn = MSELoss()
    for epoch in range(epochs):
        for i, (X, y) in enumerate(dl):
            X_with_const = torch.cat([X, torch.ones(size=(X.size()[0], 1))], dim=1)
            # forward
            y_hat = torch.einsum('bi,ij->bj', X_with_const,A)
            # loss
            loss = mse_loss_fn(y, y_hat)
            print(loss.item())
            # opt
            A_ls = torch.linalg.lstsq(X_with_const, y).solution
            # update
            A = alpha * A_ls + (1 - alpha) * A
