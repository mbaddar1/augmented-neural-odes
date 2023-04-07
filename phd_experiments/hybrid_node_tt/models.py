import logging
from typing import List, Tuple

import torch

from phd_experiments.hybrid_node_tt.basis import Basis
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchOdeSolver


class TensorTrainFixedRank(torch.nn.Module):
    def __init__(self, dims: List[int], fixed_rank: int, requires_grad: bool, unif_low: float, unif_high: float,
                 poly_deg: int):
        """
        dims : dims[0]-> output dim
        dims[1:order] -> input dims
        """
        super().__init__()
        self.poly_deg = poly_deg
        order = len(dims)
        tensor_uniform = torch.distributions.Uniform(low=unif_low, high=unif_high)
        # https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html
        self.core_tensors = torch.nn.ParameterDict(
            {'G0': torch.nn.Parameter(tensor_uniform.sample(torch.Size([dims[1], fixed_rank])))})
        self.out_dim = dims[0]
        for i in range(2, order - 1):
            self.core_tensors[f"G{i - 1}"] = torch.nn.Parameter(
                tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[i], fixed_rank])))

        self.core_tensors[f"G{order - 2}"] = torch.nn.Parameter(
            tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[order - 1], self.out_dim])),
            requires_grad=requires_grad)
        assert len(self.core_tensors) == order - 1, \
            f"# core tensors should == order-1 : {len(self.core_tensors)} != {order - 1}"

    def norm(self):
        tot_norm = 0

        for tensor in self.core_tensors.values():
            tot_norm += torch.norm(tensor)
        return tot_norm

    def display_sizes(self):
        sizes = []
        for tensor in self.core_tensors:
            sizes.append(tensor.size())
        self.logger.info(f'TensorTrain sizes = {sizes}')

    def is_parameter(self):
        requires_grads = list(map(lambda x: x.requires_grad, self.core_tensors))
        return all(requires_grads)

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        Phi = Basis.poly(z=z, t=t, poly_deg=self.poly_deg)
        assert len(self.core_tensors) == len(
            Phi), f"# of core-tensors must == number of basis tensors " \
                  f", {len(self.core_tensors)} != {len(Phi)}"
        n_cores = len(self.core_tensors)
        # first core
        core = self.core_tensors[f"G{0}"]
        res_tensor = torch.einsum("ij,bi->bj", core, Phi[0])
        # middle cores
        for i in range(1, len(self.core_tensors) - 1):
            core = self.core_tensors[f"G{i}"]
            core_basis = torch.einsum("ijk,bj->bik", core, Phi[i])
            res_tensor = torch.einsum("bi,bik->bk", res_tensor, core_basis)
        # last core
        core = self.core_tensors[f"G{n_cores - 1}"]
        core_basis = torch.einsum("ijl,bj->bil", core, Phi[n_cores - 1])
        res_tensor = torch.einsum("bi,bil->bl", res_tensor, core_basis)
        assert res_tensor.size()[1] == self.out_dim, f"output tensor size must = " \
                                                     f"out_dim : {res_tensor.size()}!={self.out_dim}"
        return res_tensor

    def display_cores(self):
        self.logger.info(f'Cores : \n{self.core_tensors}\n')


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    main_logger = logging.getLogger()
    order = 4
    basis_dim = 3
    out_dim = 10
    fixed_rank = 2

    ttxr = TensorTrainFixedRank(order=order, core_input_dim=basis_dim, out_dim=out_dim, fixed_rank=fixed_rank,
                                requires_grad=True)
    norm_val = ttxr.norm()
    ttxr.display_sizes()
    ttxr.is_parameter()
    main_logger.info(ttxr.is_parameter())

    basis_ = []
    for i in range(order):
        basis_.append(torch.ones(basis_dim))
    res = ttxr.contract_basis(basis_tensors=basis_)


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

    def __init__(self, Dz: int, basis_type: str, unif_low: float, unif_high: float,
                 tt_rank: str | int | List[int], poly_deg: int):
        super().__init__()
        # check params
        assert isinstance(poly_deg, int), f"deg must be int, got {type(self.deg)}"
        assert isinstance(tt_rank, int), f"Supporting fixed ranks only"
        assert basis_type == "poly", f"Supporting only poly basis"

        dims_A = [Dz] + [poly_deg + 1] * (Dz + 1)  # deg+1 as polynomial start from z^0 , z^1 , .. z^deg
        # Dz+1 to append time
        self.order = len(dims_A)
        self.A_TT = TensorTrainFixedRank(dims=dims_A, fixed_rank=tt_rank, unif_low=unif_low, unif_high=unif_high,
                                         requires_grad=True, poly_deg=poly_deg)

    def forward(self, t: float, z: torch.Tensor):
        """
        dzdt = A.Phi([z,t])
        """
        # dzdt = torch.nn.ReLU()(self.A_TT(t, z, self.deg))
        dzdt = self.A_TT(t, z)
        return dzdt


class LearnableOde(torch.nn.Module):
    ACTIVATIONS = (torch.nn.Identity, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.ReLU)

    def __init__(self, Dx: int, Dz: int, Dy: int, solver: TorchOdeSolver, t_span: Tuple, tensor_dtype: torch.dtype,
                 unif_low: float, unif_high: float, ode_func: torch.nn.Module,
                 output_activation: torch.nn.Module, output_linear_learnable: bool, projection_learnable: bool):
        super().__init__()
        self.output_activation = output_activation
        self.ode_func = ode_func
        self.unif_high = unif_high
        self.unif_low = unif_low
        self.tensor_dtype = tensor_dtype
        self.solver = solver
        self.t_span = t_span
        self.Dy = Dy
        self.Dx = Dx
        self.Dz = Dz
        ##
        self.check_params()
        self.P = torch.nn.Parameter(torch.distributions.Uniform(unif_low, unif_high).sample(torch.Size([Dx, Dz])),
                                    requires_grad=projection_learnable)
        output_linear = torch.nn.Linear(Dz, Dy)
        output_linear.weight.requires_grad = output_linear_learnable
        output_linear.bias.requires_grad = output_linear_learnable

        self.Q = torch.nn.Sequential(self.output_activation, output_linear)

    def forward(self, x: torch.Tensor):
        z0 = torch.einsum('bi,ij->bj', x, self.P)
        soln = self.solver.solve_ivp(func=self.ode_func, t_span=self.t_span, z0=z0, args=None)
        zT = soln.z_trajectory[-1]
        y_hat = self.Q(zT)
        return y_hat

    def check_params(self):
        assert self.Dz >= self.Dx, "latent-dim must be >= input-dima"
        assert isinstance(self.output_activation,
                          LearnableOde.ACTIVATIONS), f"output layer non-linearity must be one of " \
                                                     f"{LearnableOde.ACTIVATIONS}"
