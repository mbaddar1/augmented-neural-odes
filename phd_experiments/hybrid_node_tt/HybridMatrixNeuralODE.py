"""
References

Stability_and_steady_state_of_the_matrix_system
https://en.wikipedia.org/wiki/Matrix_differential_equation#Stability_and_steady_state_of_the_matrix_system
ODE and PDE Stability Analysis
https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture19_pde2.pdf
"""
import logging
import random
from enum import Enum
from typing import Any, Callable
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import torch.nn
from torch.nn import Sequential, MSELoss

from phd_experiments.datasets.custom_dataset import CustomDataSet
from phd_experiments.datasets.torch_boston_housing import TorchBostonHousingPrices
from phd_experiments.datasets.toy_ode import ToyODE
from phd_experiments.datasets.toy_relu import ToyRelu
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

"""
Model Design
1. Z0 = X
2. Linear ode model that takes Z0 to ZT via dz/dt = A.z , not basis
3. NN that compresses ZT to y_hat
Linear ODE from
"""


class OptMethod(Enum):
    MATRIX_LEAST_SQUARES = 0
    GRADIENT_DESCENT = 1


class OdeFuncType(Enum):
    NN = 0
    MATRIX = 1


class DataSetInstance(Enum):
    TOY_ODE = 0
    TOY_RELU = 1
    BOSTON_HOUSING = 2


class OdeFuncMatrix(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.A = torch.nn.Parameter(
            torch.distributions.Uniform(low=0.01, high=0.05).sample(sample_shape=torch.Size([input_dim, latent_dim])))

    def forward(self, t, y):
        dydt = torch.einsum('bi,ij->bj', y ** 3, self.A)
        return dydt

    def set_A(self, A_new):
        self.A = torch.nn.Parameter(A_new)


class OdeFuncNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OdeFuncNN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


class EulerFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, params: dict) -> Any:
        step_size = params['step_size']
        t_span = params['t_span']
        ode_func = params['ode_func']
        solver = TorchEulerSolver(step_size)
        # solver = TorchRK45(device=torch.device('cpu'))
        soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=x)
        ctx.z_traj = soln.z_trajectory
        ctx.t_vals = soln.t_values
        zT = soln.z_trajectory[-1]
        ctx.ode_func = ode_func
        return zT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        lr = 0.01
        alpha = 0.8
        A_old = ctx.ode_func.A
        zT = ctx.z_traj[-1]
        dLdzT = grad_outputs[0]
        zT_prime = zT - lr * dLdzT
        z_traj_new = ctx.z_traj
        z_traj_new[-1] = zT_prime
        delta_z = pd.Series(data=z_traj_new).diff(1)[1:].values
        delta_t = np.nanmean(pd.Series(data=ctx.t_vals).diff(1)[1:].values)  # assume fixed delta-t
        X_ls = torch.concat(tensors=ctx.z_traj[:-1])
        Y_ls = torch.concat(tensors=list(delta_z / delta_t))
        try:
            ls_soln = torch.linalg.lstsq(X_ls, Y_ls)
        except Exception as e:
            x = 10
        A_ls = ls_soln.solution
        A_new = alpha * A_ls + (1 - alpha) * A_old
        ctx.ode_func.set_A(A_new)
        return None, None


class HybridMatrixNeuralODE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, nn_hidden_dim, output_dim, opt_method: str, ode_func: Callable):
        super().__init__()
        self.opt_method = opt_method
        self.ode_func = ode_func
        self.Q = Sequential(torch.nn.Linear(latent_dim, nn_hidden_dim), torch.nn.Tanh(),
                            torch.nn.Linear(nn_hidden_dim, output_dim))

    def forward(self, x):
        t_span = 0, 1
        step_size = 0.02
        solver = TorchEulerSolver(step_size=step_size)
        if self.opt_method == OptMethod.MATRIX_LEAST_SQUARES:
            fn = EulerFunc()
            self.params = {'t_span': t_span, 'ode_func': self.ode_func, 'step_size': step_size,
                           'solver': solver}
            zT = fn.apply(x, self.params)
        elif self.opt_method == OptMethod.GRADIENT_DESCENT:
            soln = solver.solve_ivp(func=self.ode_func, t_span=t_span, z0=x)
            zT = soln.z_trajectory[-1]
        else:
            raise ValueError(f'Unknown opt method = {self.opt_method}')
        y_hat = self.Q(zT)
        return y_hat


def get_dataset(dataset_instance: Enum, N: int = 2024, input_dim: int = None, output_dim: int = None) -> CustomDataSet:
    if dataset_instance == DataSetInstance.TOY_ODE:
        return ToyODE(N)
    elif dataset_instance == DataSetInstance.TOY_RELU:
        return ToyRelu(N=N, input_dim=input_dim, out_dim=output_dim)
    elif dataset_instance == DataSetInstance.BOSTON_HOUSING:
        return TorchBostonHousingPrices(csv_file="../datasets/boston.csv")
    else:
        raise ValueError(f'dataset-name is not known {dataset_instance}')


def get_ode_func(ode_func_type: Enum, input_dim: int, nn_hidden_dim: int, latent_dim: int,
                 output_dim: int) -> torch.nn.Module:
    """
    some params are useless, passing all for consistency
    """
    if ode_func_type == OdeFuncType.NN:
        return OdeFuncNN(input_dim=input_dim, hidden_dim=nn_hidden_dim, output_dim=output_dim)
    elif ode_func_type == OdeFuncType.MATRIX:
        return OdeFuncMatrix(input_dim=input_dim, latent_dim=latent_dim)
    else:
        raise ValueError(f'Unknown ode func type {ode_func_type}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # configs
    epochs = 10000
    batch_size = 128
    lr = 1e-3
    nn_hidden_dim = 50
    # TODO debug boston experiment
    opt_method = OptMethod.MATRIX_LEAST_SQUARES
    dataset_instance = DataSetInstance.BOSTON_HOUSING
    train_size_ratio = 0.8
    ode_func_type = OdeFuncType.NN
    N = 2024
    input_dim = 3
    output_dim = 1
    ##
    overall_dataset = get_dataset(dataset_instance=dataset_instance, N=N, input_dim=input_dim, output_dim=output_dim)
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    latent_dim = input_dim
    # split data to train_test sets
    splits = random_split(dataset=overall_dataset, lengths=[train_size_ratio, 1 - train_size_ratio])
    train_dataset = splits[0]
    test_dataset = splits[1]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # define ode
    ode_func = get_ode_func(ode_func_type=ode_func_type, input_dim=input_dim, output_dim=output_dim,
                            nn_hidden_dim=nn_hidden_dim,
                            latent_dim=latent_dim)

    model = HybridMatrixNeuralODE(latent_dim=latent_dim, input_dim=input_dim, nn_hidden_dim=nn_hidden_dim,
                                  output_dim=output_dim,
                                  opt_method=opt_method,
                                  ode_func=ode_func)
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    loss_fn = MSELoss()
    epochs_avg_losses = []
    logger.info(f"Optimization method = {opt_method}")
    for epoch in range(epochs):
        batches_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y, y_hat)
            batches_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epochs_avg_losses.append(np.nanmean(batches_losses))
        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch} Avg-mse-Loss = {epochs_avg_losses[-1]}')
