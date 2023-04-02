import string
from enum import Enum

import torch

from phd_experiments.datasets.custom_dataset import CustomDataSet
from phd_experiments.datasets.torch_boston_housing import TorchBostonHousingPrices
from phd_experiments.datasets.toy_ode import ToyODE
from phd_experiments.datasets.toy_relu import ToyRelu
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


def generate_einsum_string(order: int):
    MAX_ORDER = 39
    chars = string.ascii_lowercase
    assert order > 0 and order % 2 == 0, "order must be > 0 and even"
    """
    let number of unique characters in einsum str = x 
    then x/2*3 must be <= 26 (number of chr) , then x <=39
    """
    assert order <= MAX_ORDER, f"order must be <= {MAX_ORDER}"
    k = int(order / 2)
    str1 = chars[:order]
    str2 = str1[k:order] + chars[order:(order + k)]
    str3 = str1[:k] + chars[order:(order + k)]
    einsum_str = f"{str1},{str2}->{str3}"
    return einsum_str


class ForwardMethod(Enum):
    EXP = 0
    INTEGRATION = 1


class SolverType(Enum):
    TORCH_EULER = 0
    TORCH_RK45 = 1


class OdeFuncType(Enum):
    NN = 0
    MATRIX = 1


class DataSetInstance(Enum):
    TOY_ODE = 0
    TOY_RELU = 1
    BOSTON_HOUSING = 2


def get_dataset(dataset_instance: Enum, N: int = 2024, input_dim: int = None, output_dim: int = None) -> CustomDataSet:
    if dataset_instance == DataSetInstance.TOY_ODE:
        return ToyODE(N)
    elif dataset_instance == DataSetInstance.TOY_RELU:
        return ToyRelu(N=N, input_dim=input_dim, out_dim=output_dim)
    elif dataset_instance == DataSetInstance.BOSTON_HOUSING:
        return TorchBostonHousingPrices(csv_file="../datasets/boston.csv")
    else:
        raise ValueError(f'dataset-name is not known {dataset_instance}')


def get_solver(solver_type: Enum, **kwargs):
    if solver_type == SolverType.TORCH_EULER:
        return TorchEulerSolver(step_size=kwargs['step_size'])
    elif solver_type == SolverType.TORCH_RK45:
        return TorchRK45(device=torch.device("cpu"), tensor_dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported solver type {solver_type}")
