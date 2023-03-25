from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class CustomDataSet(Dataset):
    def __getitem__(self, index) -> T_co:
        pass

    @abstractmethod
    def get_input_dim(self):
        pass

    @abstractmethod
    def get_output_dim(self):
        pass
