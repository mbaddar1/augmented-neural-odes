import torch
from torch.utils.data import Dataset

from phd_experiments.datasets.custom_dataset import CustomDataSet


class ToyRelu(CustomDataSet):

    def __init__(self, N, input_dim, out_dim):
        self.N = N
        self.input_dim = input_dim
        self.output_dim = out_dim
        self.X = torch.randn((self.N, input_dim))
        self.y = (torch.rand(size=(self.N, out_dim)) < 0.5).float()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim
