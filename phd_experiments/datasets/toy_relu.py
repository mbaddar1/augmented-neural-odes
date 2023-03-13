import torch
from torch.utils.data import Dataset


class ToyRelu(Dataset):
    def __init__(self, N, input_dim, out_dim):
        self.N = N
        self.data_x = torch.randn((self.N, input_dim))
        self.data_y = (torch.rand(size=(self.N, out_dim)) < 0.5).float()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
