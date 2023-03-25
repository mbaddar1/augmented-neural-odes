"""
https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L34
"""
import torch.nn
from torch.utils.data import Dataset

from phd_experiments.datasets.custom_dataset import CustomDataSet
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


class TrueODEFunc(torch.nn.Module):
    def __init__(self, true_A):
        super().__init__()
        self.true_A = true_A

    def forward(self, t, y):
        dydt = torch.einsum('bi,ij->bj', y ** 3, self.true_A)
        return dydt


class ToyODE(CustomDataSet):
    def __init__(self, N: int):
        super().__init__()
        self.N = N
        self.input_dim = 2
        self.output_dim = 2
        dtype = torch.float32
        device = torch.device("cpu")
        t_span = 0, 1
        delta_t = 0.1
        solver = TorchEulerSolver(step_size=delta_t)
        # solver = TorchRK45(device=torch.device("cpu"),tensor_dtype=dtype)
        true_y0 = torch.distributions.MultivariateNormal(loc=torch.tensor([2.0, -1.0]),
                                                         scale_tril=torch.diag(torch.tensor([0.01, 0.01]))).sample(
            torch.Size([self.N]))
        true_A = torch.tensor([[-0.1, 0.8], [-0.9, -0.1]]).to(device)
        true_ode_func = TrueODEFunc(true_A=true_A)
        soln = solver.solve_ivp(func=true_ode_func, t_span=t_span, z0=true_y0)
        yT = soln.z_trajectory[-1]
        self.X = true_y0
        self.Y = yT

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.N

    @staticmethod
    def ode_func(t, y: torch.Tensor, true_A: torch.Tensor) -> torch.Tensor:
        dydt = torch.einsum("bi,ij->bj", y ** 3, true_A)
        return dydt

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.output_dim
