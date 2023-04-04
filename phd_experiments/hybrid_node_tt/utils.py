import string
from enum import Enum
from typing import List
import torch
from phd_experiments.datasets.custom_dataset import CustomDataSet
from phd_experiments.datasets.torch_boston_housing import TorchBostonHousingPrices
from phd_experiments.datasets.toy_ode import ToyODE
from phd_experiments.datasets.toy_relu import ToyRelu
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


def generate_tensor_poly_einsum(dims: List[int]):
    chars = string.ascii_letters
    chars = chars.replace("b", "")  # reserve b for batch
    order = len(dims)
    k = int(order / 2)
    assert order <= len(chars) - 1, f"order must be <={len(chars) - 1}"
    assert_dims_symmetry(dims), "symmetry dims failed"
    einsum_str = chars[:order]
    for i in range(k):
        einsum_str += f',b{chars[i + k]}'
    einsum_str += f"->b{chars[:k]}"
    return einsum_str


def prod_list(l: List):
    prod_ = 1
    for e in l:
        prod_ *= e
    return prod_


def assert_dims_symmetry(dims: List[int]):
    order = len(dims)
    assert order > 0 and order % 2 == 0, f"order must be even and > 0 , got order = {order}"
    k = int(order / 2)
    for i in range(k):
        assert dims[i] == dims[i + k], f"dims[{i + 1}] = {dims[i]} != dims{[i + k]} = {dims[i + k]}"
    return True


def generate_identity_tensor(dims: List[int]):
    order = len(dims)
    k = int(order / 2)
    matricized_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
    assert matricized_dims[0] == matricized_dims[1], f"matricized dims are not symmetric : {matricized_dims}"
    matrix_identity = torch.eye(matricized_dims[0], dtype=torch.float64)
    identity_tensor = torch.reshape(matrix_identity, dims)
    return identity_tensor


def generate_einsum_string_tensor_power(order: int):
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


def get_dataset(dataset_instance: Enum, N: int = 2024, input_dim: int = 6, output_dim: int = 1) -> CustomDataSet:
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


if __name__ == '__main__':
    dims = [3]*26
    einsum_str = generate_tensor_poly_einsum(dims)
    x = 10
