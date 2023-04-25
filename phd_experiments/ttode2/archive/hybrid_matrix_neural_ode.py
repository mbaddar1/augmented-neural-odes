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
from typing import Any, Callable, Tuple
import pandas as pd
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import torch.nn
from torch.nn import Sequential, MSELoss

from phd_experiments.ttode2.utils import OdeFuncType, DataSetInstance, get_dataset, get_solver, SolverType, \
    ForwardMethod
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchOdeSolver

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


class OdeFuncMatrix(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # self.A = torch.nn.Parameter(
        #     torch.distributions.Uniform(low=0.01, high=0.05).sample(sample_shape=torch.Size([input_dim, latent_dim])))
        # https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
        # A = S.Lambda.S_inv
        self.S = torch.nn.Parameter(
            torch.distributions.Uniform(0.01, 0.02).sample(torch.Size([latent_dim, latent_dim])))
        lambda_diag = torch.distributions.Uniform(0.2, 0.4).sample(torch.Size([latent_dim]))
        self.Lambda = torch.nn.Parameter(torch.diag(lambda_diag))
        x = 10
        # self.A = torch.nn.Parameter(
        #     torch.diag(torch.distributions.Uniform(-0.2, -0.1).sample(torch.Size([latent_dim]))))

    def forward(self, t, y):
        M = torch.matmul(self.S, self.Lambda)
        A = torch.matmul(M, torch.linalg.pinv(self.S))
        eval = torch.linalg.eigvals(A)
        dydt = torch.einsum('bi,ij->bj', y ** 3, A)
        return dydt

    # def set_A(self, A_new):
    #     self.A = torch.nn.Parameter(A_new)


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
    def forward(ctx: Any, x: torch.Tensor, P: torch.Tensor, params: dict) -> Any:
        step_size = params['step_size']
        t_span = params['t_span']
        ode_func = params['ode_func']
        ctx.lr = params['lr']
        ctx.alpha = params['alpha']
        solver = TorchEulerSolver(step_size)
        # solver = TorchRK45(device=torch.device('cpu'))
        z0 = torch.einsum('bi,ij->bj', x, P)
        eigenvalues, eigenvectors = torch.linalg.eig(ode_func.A)
        soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=z0)
        ctx.z_traj = soln.z_trajectory
        ctx.t_vals = soln.t_values
        zT = soln.z_trajectory[-1]
        ctx.ode_func = ode_func
        ctx.P = P
        for z in soln.z_trajectory:
            if torch.any(torch.isinf(z)) or torch.any(torch.isnan(z)):
                raise ValueError(f'ZT has inf, the forward-integrate had exploded!!!!')
        return zT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        zT = ctx.z_traj[-1]
        dLdzT = grad_outputs[0]
        lr = 0.01
        zT_prime = zT - lr * dLdzT
        z_traj_new = ctx.z_traj
        z_traj_new[-1] = zT_prime
        traj_len = len(z_traj_new)
        z_traj_new = z_traj_new[traj_len - 3:traj_len - 1]
        delta_z = pd.Series(data=z_traj_new).diff(1)[1:].values
        delta_t = np.nanmean(pd.Series(data=ctx.t_vals).diff(1)[1:].values)  # assume fixed delta-t
        X_ls = z_traj_new[0]  # torch.concat(tensors=ctx.z_traj[:-1])
        Y_ls = torch.concat(tensors=list(delta_z / delta_t))
        try:
            ls_soln = torch.linalg.lstsq(X_ls, Y_ls)
        except Exception as e:
            raise Exception(f"exception at lstsq = {e}")
        A_ls = ls_soln.solution
        eval = torch.linalg.eigvals(A_ls)
        # TODO add eigen-val check
        ctx.ode_func.set_A(A_ls)
        return None, None, None


class HybridMatrixNeuralODE(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, opt_method: Enum, ode_func_nn: Callable,
                 lr: float,
                 alpha: float, solver: TorchOdeSolver, forward_method: Enum, t_span: Tuple):
        super().__init__()
        self.opt_method = opt_method
        self.ode_func_nn = ode_func_nn
        self.S = torch.nn.Parameter(
            torch.distributions.Uniform(0.01, 0.02).sample(torch.Size([latent_dim, latent_dim])))
        self.lambda_vals = torch.nn.Parameter(torch.distributions.Uniform(0.2, 0.6).sample(torch.Size([latent_dim])))
        self.Q = Sequential(torch.nn.Tanh(), torch.nn.Linear(latent_dim, output_dim))
        self.lr = lr
        self.alpha = alpha
        self.solver = solver
        self.t_span = t_span
        self.forward_method = forward_method

    def forward(self, x):
        z0 = x
        if self.forward_method == ForwardMethod.EXP:
            delta_t = self.t_span[1] - self.t_span[0]
            exp_lambda = torch.exp(self.lambda_vals * delta_t)
            M = torch.matmul(self.S, torch.diag(exp_lambda))
            exp_At = torch.matmul(M, torch.linalg.pinv(self.S))

            zT_transpose = torch.einsum('dd,db->db', exp_At, z0.T)
            zT = zT_transpose.T
        elif self.forward_method == ForwardMethod.INTEGRATION:

            soln = self.solver.solve_ivp(func=self.ode_func_nn, t_span=self.t_span, z0=z0)
            zT = soln.z_trajectory[-1]
        else:
            raise ValueError(f"Unimplemented forward method {self.forward_method}")

        y_hat = self.Q(zT)
        return y_hat


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
    N = 4096
    epochs = 20000
    batch_size = 256
    lr = 1e-3
    nn_hidden_dim = 50
    alpha = 1.0
    data_loader_shuffle = False
    # TODO debug boston experiment
    opt_method = OptMethod.GRADIENT_DESCENT
    dataset_instance = DataSetInstance.BOSTON_HOUSING
    solver_type = SolverType.TORCH_RK45
    forward_method = ForwardMethod.INTEGRATION
    t_span = 0, 0.8
    train_size_ratio = 0.8
    input_dim = 3
    output_dim = 1
    stop_thr = 1e-2

    ##
    overall_dataset = get_dataset(dataset_instance=dataset_instance, N=N, input_dim=input_dim,
                                  output_dim=output_dim)
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    latent_dim = input_dim
    # split data to train_test sets
    splits = random_split(dataset=overall_dataset, lengths=[train_size_ratio, 1 - train_size_ratio])
    train_dataset = splits[0]
    test_dataset = splits[1]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)
    # define ode
    ode_func_nn = OdeFuncNN(input_dim=input_dim, hidden_dim=nn_hidden_dim, output_dim=output_dim)
    solver = get_solver(solver_type=solver_type)
    model = HybridMatrixNeuralODE(latent_dim=latent_dim,
                                  output_dim=output_dim,
                                  opt_method=opt_method,
                                  ode_func_nn=ode_func_nn, lr=lr,
                                  alpha=alpha, solver=solver,
                                  forward_method=forward_method, t_span=t_span)
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    loss_fn = MSELoss()
    epochs_avg_losses = []

    # logging about the problem and opt. method
    logger.info(f"Dataset = {dataset_instance}")
    logger.info(f"Optimization method = {opt_method}")
    for epoch in range(epochs):
        batches_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y, y_hat)  # + 0.1 * torch.norm(ode_func.A) # + slack_var
            batches_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epochs_avg_losses.append(np.nanmean(batches_losses))
        if epoch > 10:
            avg_last_diff = np.nanmean(pd.Series(data=epochs_avg_losses).diff(1).
                                       apply(lambda x: abs(x)).values[1:])
        if epoch % 10 == 0:
            logger.info(f'Epoch {epoch} Avg-mse-Loss = {epochs_avg_losses[-1]}')
