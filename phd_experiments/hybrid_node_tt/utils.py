import logging
import os
import string
from datetime import datetime
from enum import Enum
from typing import List
import torch
from torch.nn import MSELoss

from phd_experiments.datasets.custom_dataset import CustomDataSet
from phd_experiments.datasets.torch_boston_housing import TorchBostonHousingPrices
from phd_experiments.datasets.toy_ode import ToyODE
from phd_experiments.datasets.toy_relu import ToyRelu
from phd_experiments.hybrid_node_tt.models import TensorTrainOdeFunc, NNodeFunc
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


def generate_tensor_poly_einsum(order: int):
    chars = string.ascii_letters
    chars = chars.replace("b", "")  # reserve b for batch
    einsum_str = chars[:order]
    for i in range(1, order):
        einsum_str += f',b{chars[i]}'
    einsum_str += f"->b{chars[0]}"
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


def get_dataset(config: dict) -> CustomDataSet:
    dataset_name = config["train"]["dataset"]
    N = config["train"]["N"]
    if dataset_name == "toy-ode":
        return ToyODE(N)
    elif dataset_name == "toy-relu":
        return ToyRelu(N=N)
    elif dataset_name == "boston-housing":
        return TorchBostonHousingPrices(csv_file="../datasets/boston.csv")
    else:
        raise ValueError(f'dataset-name is not known {dataset_name}')


def get_solver(config: dict):
    if config["ode"]["solver"]["method"] == "euler":
        return TorchEulerSolver(step_size=config["ode"]["solver"]["euler"]['step-size'])
    elif config["solver"]["method"] == "rk45":
        return TorchRK45(device=torch.device(config["train"]["device"]), tensor_dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported solver type {config['solver']['method']}")


def get_ode_func(config: dict):
    assert config["init"]["method"] == "uniform", "now only support unif. init. "
    init_method = config["init"]["method"]
    if config["ode"]["model"] == "tt":
        assert config["ode"]["tt"]["rank"].isnumeric(), "Supporting fixed rank now only"
        tt_rank = int(config["ode"]["tt"]["rank"])
        return TensorTrainOdeFunc(Dz=config["container"]["latent-dim"],
                                  basis_model=config["ode"]["tt"]["basis"]["model"],
                                  unif_low=config["init"][init_method]["low"],
                                  unif_high=config["init"][init_method]["high"],
                                  tt_rank=tt_rank, poly_deg=config['ode']['tt']['basis']['poly']['deg'])
    elif config["ode"]["model"] == "nn":
        return NNodeFunc(latent_dim=config["container"]["latent-dim"], nn_hidden_dim=config["ode"]["nn"]["hidden-dim"])
    else:
        raise ValueError(f"""Unsupported ode-func model {config["ode-func"]["model"]}""")


def get_tensor_dtype(config: dict):
    if config['train']['dtype'] == "torch.float32":
        return torch.float32
    elif config['train']['dtype'] == "torch.float64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported tensor type = {config['train']['dtype']}")


def get_activation(activation_name: str):
    ACTIVATION_DICT = {'tanh': torch.nn.Tanh(), 'relu': torch.nn.ReLU(), 'sigmoid': torch.nn.Sigmoid(),
                       'identity': torch.nn.Identity()}
    activation = ACTIVATION_DICT.get(activation_name, None)
    if activation is None:
        raise ValueError(f"Unknown activation name {activation_name}")
    else:
        return activation


def get_loss_function(loss_name: str):
    if loss_name == "mse":
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss :{loss_name}")


def get_logger(level: str, date_time_format: str, log_format: str, experiments_counter_file_path: str,
               experiments_log_dir: str):
    LOG_LEVEL_MAP = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                     'error': logging.WARNING}
    log_level_enum = LOG_LEVEL_MAP.get(level, None)
    assert log_level_enum is not None, f"Unknown log-level {level}"
    tstamp = datetime.now().strftime(date_time_format)
    logging.basicConfig(level=log_level_enum, format=log_format)
    logger = logging.getLogger()
    with open(experiments_counter_file_path, "r") as f:
        experiment_number = int(f.readline()) + 1
        f.close()
    with open(experiments_counter_file_path, "w") as f:
        f.write(str(experiment_number))
        f.flush()
        f.close()
    log_file_path = os.path.join(experiments_log_dir, f"experiment_no_{experiment_number}_{tstamp}.log")
    # set handlers
    formatter = logging.Formatter(fmt=log_format)
    fh = logging.FileHandler(filename=log_file_path, mode="w")
    fh.setLevel(level=log_level_enum)
    fh.setFormatter(fmt=formatter)
    logger.addHandler(fh)
    return logger,experiment_number


def assert_models_learnability(config: dict, projection_model: torch.nn.Module, output_model: torch.nn.Module):
    assert projection_model.is_learnable() == config['projection']['learnable'], \
        f"projection-model learnability not as configured"
    assert output_model.is_learnable() == config['output']['learnable'], \
        f"output-model learnability not as configured"
