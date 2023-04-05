from collections import OrderedDict

from phd_experiments.hybrid_node_tt.tt2 import TensorTrainFixedRank
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchOdeSolver
import logging
import random
from typing import Tuple, List
import numpy as np
import torch.nn
from torch.nn import MSELoss
from torch.utils.data import random_split, DataLoader
from phd_experiments.hybrid_node_tt.basis import Basis
from phd_experiments.hybrid_node_tt.utils import DataSetInstance, get_dataset, generate_tensor_poly_einsum
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

"""
stability of linear ode
https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2010/2010_4.2%20Stability%20and%20Linearization%20of%20ODEs.pdf
https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture19_pde2.pdf
https://faculty.ksu.edu.sa/sites/default/files/stability.pdf
"""

"""
Objective of this script is to verify on research hypothesis ( Tensor-Neural ODE expressive power , no focus on
memory or speed 
dzdt = A.phi([z,t]) that works with complex problems
"""


class TensorTrainOdeFunc(torch.nn.Module):
    def __init__(self, Dz: int, basis_type: str, basis_param: dict, unif_low: float, unif_high: float,
                 tt_rank: str | int | List[int]):
        super().__init__()
        self.deg = basis_param.get("deg")
        # check params
        assert isinstance(self.deg, int), f"deg must be int, got {type(self.deg)}"
        assert isinstance(tt_rank, int), f"Supporting fixed ranks only"
        assert basis_type == "poly", f"Supporting only poly basis"
        dims_A = [Dz] + [self.deg + 1] * (Dz + 1)  # deg+1 as polynomial start from z^0 , z^1 , .. z^deg
        # Dz+1 to append time
        self.order = len(dims_A)
        self.A_TT = TensorTrainFixedRank(dims=dims_A, fixed_rank=tt_rank, unif_low=unif_low, unif_high=unif_high,
                                         requires_grad=True)

    def forward(self, t: float, z: torch.Tensor):
        """
        dzdt = A.Phi([z,t])
        """
        dzdt = self.A_TT(t, z, self.deg)
        return dzdt


class Qnn(torch.nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        linear_part = torch.nn.Linear(latent_dim, output_dim)
        linear_part.weight.requires_grad = False
        linear_part.bias.requires_grad = False
        # ('non-linearity', torch.nn.Identity()),
        self.model = torch.nn.Sequential(OrderedDict([('output-matrix', linear_part)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        return y_hat


class HybridTensorTrainNeuralODE(torch.nn.Module):
    BASIS_TYPES = ["poly"]

    def __init__(self, Dx: int, Dz: int, Dy: int, basis_type: str, basis_params: dict,
                 solver: TorchOdeSolver, t_span: Tuple, tensor_dtype: torch.dtype, unif_low: float,
                 unif_high: float, tt_rank: str | int | List[int]):
        super().__init__()
        self.tt_rank = tt_rank
        self.unif_high = unif_high
        self.unif_low = unif_low
        self.tensor_dtype = tensor_dtype
        self.solver = solver
        self.t_span = t_span
        self.basis_params = basis_params
        self.basis_type = basis_type
        self.Dy = Dy
        self.Dx = Dx
        self.Dz = Dz
        ##
        self.check_params()
        self.P = torch.nn.Parameter(torch.distributions.Uniform(unif_low, unif_high).sample(torch.Size([Dx, Dz])),
                                    requires_grad=False)
        self.Q = Qnn(latent_dim=latent_dim, output_dim=output_dim)
        self.ode_func_tensor = TensorTrainOdeFunc(Dz=self.Dz, basis_type=self.basis_type, basis_param=self.basis_params,
                                                  unif_low=unif_low,
                                                  unif_high=unif_high, tt_rank=self.tt_rank)

    def forward(self, x: torch.Tensor):
        z0 = x  # torch.einsum("bi,ij->bj", x, self.P)
        soln = self.solver.solve_ivp(func=self.ode_func_tensor, t_span=self.t_span, z0=z0, args=None)
        zT = soln.z_trajectory[-1]
        # y_hat = self.Q(zT)
        return zT

    def check_params(self):
        assert self.basis_type in HybridTensorTrainNeuralODE.BASIS_TYPES, \
            f"Unknown basis_type {self.basis_type}, must be on of {HybridTensorTrainNeuralODE.BASIS_TYPES}"
        if self.basis_type == "poly":
            assert "deg" in self.basis_params.keys() and isinstance(self.basis_params["deg"], int), \
                f"As basis_type is poly, basis-params must contain deg param as an int"
        assert isinstance(self.tt_rank, int), f"supporting fixed tt rank ( int) , got {type(self.tt_rank)}"


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # configs
    N = 4096
    epochs = 200
    batch_size = 256
    lr = 1e-3
    data_loader_shuffle = False
    dataset_instance = DataSetInstance.TOY_ODE
    t_span = 0, 1
    train_size_ratio = 0.8
    poly_deg = 3
    basis_type = "poly"
    device = torch.device("cpu")
    tensor_dtype = torch.float32
    unif_low, unif_high = 0.01, 0.05
    fixed_tt_rank = 5
    # get dataset and loader
    overall_dataset = get_dataset(dataset_instance=dataset_instance, N=N)
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    splits = random_split(dataset=overall_dataset, lengths=[train_size_ratio, 1 - train_size_ratio])
    train_dataset = splits[0]
    test_dataset = splits[1]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)

    # create model
    latent_dim = input_dim
    assert latent_dim == input_dim
    solver = TorchRK45(device=device, tensor_dtype=tensor_dtype)
    model = HybridTensorTrainNeuralODE(Dx=input_dim, Dz=latent_dim, Dy=output_dim, basis_type=basis_type,
                                       basis_params={'deg': poly_deg}, solver=solver, t_span=t_span,
                                       tensor_dtype=tensor_dtype, unif_low=unif_low, unif_high=unif_high,
                                       tt_rank=fixed_tt_rank)
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        batches_loss = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            batches_loss.append(loss.item())
            # TODO add check point to get A-cores before optimization
            A_norm_before = model.ode_func_tensor.A_TT.norm()
            loss.backward()
            optimizer.step()
            # TODO add check point to get A-cores after optimization
            A_norm_after = model.ode_func_tensor.A_TT.norm()
            delta_A_norm = torch.norm(A_norm_after - A_norm_before)
            if delta_A_norm.item() > 0:
                u = 0
            x = 10
        if epoch % 10 == 0:
            print(f'At epoch = {epoch} -> avg-batches-loss = {np.nanmean(batches_loss)}')
