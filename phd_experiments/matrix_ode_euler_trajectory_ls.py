# https://jax.readthedocs.io/en/latest/installation.html
import time
from typing import Any, Callable, Tuple

import mygrad
import numpy as np
import jax.numpy as jnp
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


class F(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.Tensor([[0.1, 0.2], [0.3, 0.4]]))
        self.delta_t = 0.01

    def forward(self, z):
        z_prime = torch.matmul(self.A, z) * self.delta_t + z
        return z_prime


def function1(z, A, delta_t):
    z_prime = jnp.matmul(A, z) * delta_t + z
    return z_prime


def ode_func(t, z, A):
    dzdt = torch.einsum('bi,ij->bj', z, A)
    return dzdt


class MatrixOdeDataset(Dataset):
    def __init__(self, D, A, N, t_span, solver):
        self.N = N
        #
        # Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        # solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=Z0.dtype)
        # soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        # ZT = soln.z_trajectory[-

        # one euler step
        unif = torch.distributions.Normal(loc=0, scale=1.0)
        # delta_t = t_span[1] - t_span[0]
        Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        ZT = soln.z_trajectory[-1]
        self.x_train = Z0
        self.y_train = ZT

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def __len__(self):
        return self.N



class MatrixODEAutoGradFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, Z0: torch.Tensor, params: dict, solver: TorchRK45, ode_func: Callable, t_span: Tuple) -> Any:
        A = params['A']
        soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        ZT = soln.z_trajectory[-1]

        # back the ctx object
        ctx.A = A
        ctx.lr = params['lr']
        ctx.ZT = ZT
        ctx.Z0 = Z0
        ctx.t_span = t_span
        ctx.params = params
        ctx.Z_traj = soln.z_trajectory
        ctx.t_values = soln.t_values
        ctx.solver = solver
        return ZT

    @staticmethod
    def get_A_LS_from_euler_trajectory(z_trajectory, t_values):
        z_series = pd.Series(z_trajectory)
        t_series = pd.Series(t_values)
        delta_z_series = z_series.diff(1)[1:]
        delta_t_series = t_series.diff(1)[1:]
        delta_z_delta_t_zip_list = list(zip(delta_z_series.values, delta_t_series.values))
        Y_list = list(map(lambda x: x[0] / x[1], delta_z_delta_t_zip_list))
        Y_ls_tensor = torch.concat(tensors=Y_list, dim=0)

        X_ls = torch.concat(tensors=z_trajectory[:-1], dim=0)
        lstsq_ret = torch.linalg.lstsq(X_ls, Y_ls_tensor)
        A_ls = lstsq_ret.solution
        return A_ls

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        lr = ctx.lr
        dLdz_T = grad_outputs[0]
        ZT = ctx.ZT
        Z_traj = ctx.Z_traj
        ZT_new = ZT - lr * dLdz_T

        if isinstance(solver, TorchEulerSolver):
            Z_traj[-1] = ZT_new
            A_lstsq = MatrixODEAutoGradFn.get_A_LS_from_euler_trajectory(z_trajectory=Z_traj, t_values=ctx.t_values)
            A_old = ctx.params['A']
            ctx.params['A'] = A_lstsq
        else:
            raise NotImplementedError(f'Backward Least-squares is not implemented for solver {type(ctx.solver)}')

        # assertion code
        # assert that the get_A_ALS code is quite accuracy, it assumes euler forward pass
        # let's see how it works with euler and not euler solvers
        A_ls_sanity_check = False
        if A_ls_sanity_check:
            Z_traj[-1] = ZT  # put back the original ZT
            A_lstsq_sanity = MatrixODEAutoGradFn.get_A_LS_from_euler_trajectory(z_trajectory=Z_traj,
                                                                                t_values=ctx.t_values)
            err = torch.norm(A_lstsq_sanity - A_old)
            print(f'A sanity check error = {err}')
            time.sleep(1)
        return None, None, None, None, None, None


if __name__ == '__main__':
    # Parameters
    D = 3
    A_ref = torch.distributions.Uniform(-1, 1).sample(sample_shape=torch.Size([3, 3]))
    N = 1024
    batch_size = 64
    t_span = 0, 1
    epochs = 1000
    delta_t = 0.2
    solver_type = 'torch-euler'
    alpha = 0.8
    # Experiment code start
    mse_loss = MSELoss()
    if solver_type == 'torch-euler':
        solver = TorchEulerSolver(step_size=delta_t)
    elif solver_type == 'torch-rk45':
        solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32)
    else:
        raise ValueError(f'Unknown solver-type = {solver_type}')

    ds = MatrixOdeDataset(D=D, A=A_ref, N=N, solver=solver, t_span=t_span)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    func = MatrixODEAutoGradFn()
    unif = torch.distributions.Uniform(0.01, 0.05)
    A_train = torch.nn.Parameter(unif.sample(torch.Size([D, D])))

    params = {'A': A_train, 'lr': 0.8}
    loss_threshold = 1e-3
    for epoch in range(1, epochs + 1):
        batches_losses = []
        mse_loss_fn = MSELoss()
        for i, (Z0, ZT) in enumerate(dl):
            A_old = params['A']
            ZT_hat = func.apply(Z0, params, solver, ode_func, t_span)
            loss = mse_loss_fn(ZT_hat, ZT)  # +0.2*torch.norm(params['A'])
            loss.backward(retain_graph=True)
            A_new = params['A']
            A_updated = alpha * A_new + (1 - alpha) * A_old
            # print(f'Norm A = {torch.norm(A_updated)}')
            # print(f'A updated = A {A_updated}')
            params['A'] = A_updated
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        if epoch % 10 == 0:
            print(f'epoch = {epoch} , loss = {epoch_avg_loss}')
        if epoch_avg_loss <= 1e-3:
            print(f'loss <= threshold = {loss_threshold}, Terminating ! ')
            break
