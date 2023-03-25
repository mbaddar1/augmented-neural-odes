import torch
from torch.utils.data import Dataset

from phd_experiments.datasets.custom_dataset import CustomDataSet


class ToyLinearDataSet1(CustomDataSet):

    def __init__(self, N: int, A: torch.Tensor, b: torch.Tensor, dist: torch.distributions.Distribution):
        self.N = N
        self.Dx = A.size()[0]
        self.Dy = A.size()[1]
        self.X = dist.sample(sample_shape=torch.Size([N, self.Dx]))
        self.Y = torch.einsum('bi,ij->bj', self.X, A) + b

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.N

    def get_input_dim(self):
        return self.Dx

    def get_output_dim(self):
        return self.Dy
