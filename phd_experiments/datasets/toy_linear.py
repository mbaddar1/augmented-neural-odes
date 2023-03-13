import torch
from torch.utils.data import Dataset


class ToyLinearDataSet1(Dataset):
    def __init__(self, N: int, A: torch.Tensor, b: torch.Tensor, dist: torch.distributions.Distribution):
        self.N = N
        Dx = A.size()[0]
        self.X = dist.sample(sample_shape=torch.Size([N, Dx]))
        self.Y = torch.einsum('bi,ij->bj', self.X, A) + b

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.N
