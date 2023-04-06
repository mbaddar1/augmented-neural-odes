from collections import OrderedDict

from phd_experiments.hybrid_node_tt.tt2 import TensorTrainFixedRank
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchOdeSolver
import logging
import random
from typing import Tuple, List
import numpy as np
import torch.nn
from torch.nn import MSELoss, Sequential
from torch.utils.data import random_split, DataLoader
from phd_experiments.hybrid_node_tt.utils import DataSetInstance, get_dataset, generate_tensor_poly_einsum
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

#
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# TODO
#   * Document 2x3 experiments (tt/nn-odefunc) x ( toy_ode,toy_relu,boston datasets)
#       - save each experiment dump in a test file under experiments log
#       - document details in the gdoc
#   * experiment different weight init. schemes and document them
#       Steps
#       - https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L122
#       - https://pytorch.org/docs/stable/nn.init.html
#       - add mechanism to experiment different initializers
#       - document results in the gdoc
#   gdoc for experiments
#   https://docs.google.com/document/d/11-13S54BK4fdqMls0yja26wtveMBZQ3G0g9krLXeyG8/edit?usp=share_link

"""
some material

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


class NNodeFunc(torch.nn.Module):
    # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L111
    def __init__(self, latent_dim, nn_hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, nn_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(nn_hidden_dim, latent_dim),
        )
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, z, *args):
        return self.net(z)


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
        # dzdt = torch.nn.ReLU()(self.A_TT(t, z, self.deg))
        dzdt = self.A_TT(t, z, self.deg)
        return dzdt


class HybridTensorTrainNeuralODE(torch.nn.Module):
    BASIS_TYPES = ["poly"]
    NON_LINEARITIES = (torch.nn.Identity, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU)

    def __init__(self, Dx: int, Dz: int, Dy: int, basis_type: str, basis_params: dict,
                 solver: TorchOdeSolver, t_span: Tuple, tensor_dtype: torch.dtype, unif_low: float,
                 unif_high: float, tt_rank: str | int | List[int], ode_func: torch.nn.Module,
                 output_non_linearity: torch.nn.Module):
        super().__init__()
        self.output_non_linearity = output_non_linearity
        self.ode_func = ode_func
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
        self.Q = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(latent_dim, output_dim))

    def forward(self, x: torch.Tensor):
        z0 = x
        soln = self.solver.solve_ivp(func=self.ode_func, t_span=self.t_span, z0=z0, args=None)
        zT = soln.z_trajectory[-1]
        y_hat = self.Q(zT)
        return y_hat

    def check_params(self):
        assert self.basis_type in HybridTensorTrainNeuralODE.BASIS_TYPES, \
            f"Unknown basis_type {self.basis_type}, must be on of {HybridTensorTrainNeuralODE.BASIS_TYPES}"
        if self.basis_type == "poly":
            assert "deg" in self.basis_params.keys() and isinstance(self.basis_params["deg"], int), \
                f"As basis_type is poly, basis-params must contain deg param as an int"
        assert isinstance(self.tt_rank, int), f"supporting fixed tt rank ( int) , got {type(self.tt_rank)}"
        assert isinstance(self.output_non_linearity,
                          HybridTensorTrainNeuralODE.NON_LINEARITIES), f"output layer non-linearity must be one of " \
                                                                       f"{HybridTensorTrainNeuralODE.NON_LINEARITIES}"


def get_tt_gradient(A_TT: TensorTrainFixedRank):
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # configs
    N = 4096
    epochs = 200
    batch_size = 128
    lr = 0.001
    data_loader_shuffle = False
    dataset_instance = DataSetInstance.TOY_ODE
    t_span = 0, 1.0
    train_size_ratio = 0.8
    poly_deg = 3
    basis_type = "poly"
    device = torch.device("cpu")
    tensor_dtype = torch.float32
    # ** Important ** Keep weights init like that, it works !!!
    # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L122

    unif_low, unif_high = -0.05, 0.05
    fixed_tt_rank = 3
    ode_func_type = "tt"
    nn_ode_func_hidden_dim = 50
    euler_step_size = 0.1
    output_non_linearity = torch.nn.Tanh()
    ######
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
    # get ode-func

    latent_dim = input_dim
    assert latent_dim == input_dim
    # TODO investigate why rk45 solver under-flows
    # solver = TorchRK45(device=device, tensor_dtype=tensor_dtype)
    solver = TorchEulerSolver(step_size=euler_step_size)
    if ode_func_type == "tt":
        ode_func = TensorTrainOdeFunc(Dz=latent_dim, basis_type=basis_type, basis_param={'deg': poly_deg},
                                      unif_low=unif_low, unif_high=unif_high, tt_rank=fixed_tt_rank)
    elif ode_func_type == "nn":
        ode_func = NNodeFunc(latent_dim=latent_dim, nn_hidden_dim=nn_ode_func_hidden_dim)
    else:
        raise ValueError(f"Unknown ode_func_type {ode_func_type}")
    model = HybridTensorTrainNeuralODE(Dx=input_dim, Dz=latent_dim, Dy=output_dim, basis_type=basis_type,
                                       basis_params={'deg': poly_deg}, solver=solver, t_span=t_span,
                                       tensor_dtype=tensor_dtype, unif_low=unif_low, unif_high=unif_high,
                                       tt_rank=fixed_tt_rank, ode_func=ode_func,
                                       output_non_linearity=output_non_linearity)
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    logger.info(f"Ode-func = {type(ode_func)}\n"
                f"Data = {type(overall_dataset)}\n"
                f"Batch-Size = {batch_size}\n"
                f"Uniform low and high = {unif_low, unif_high}\n"
                f"t_span = {t_span}")
    # TODO
    for epoch in range(1, epochs + 1):
        batches_loss = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            residual = loss_fn(y_hat, y)
            loss = residual
            batches_loss.append(residual.item())
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            logger.info(f'epoch = {epoch} -> avg-batches-loss = {np.nanmean(batches_loss)}')
