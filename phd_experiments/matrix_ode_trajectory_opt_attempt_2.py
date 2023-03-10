# https://jax.readthedocs.io/en/latest/installation.html
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
    def __init__(self, D, A, N, t_span, delta_t):
        self.N = N
        #
        # Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        # solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=Z0.dtype)
        # soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        # ZT = soln.z_trajectory[-

        # one euler step
        unif = torch.distributions.Uniform(-1, 1)
        # delta_t = t_span[1] - t_span[0]
        Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        solver = TorchEulerSolver(step_size=delta_t)
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
    def forward(ctx: Any, Z0: torch.Tensor, params: dict, solver: TorchRK45, ode_func: Callable, t_span: Tuple,
                delta_t: float) -> Any:
        A = params['A']
        solver = TorchEulerSolver(step_size=delta_t)
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
        return ZT

    @staticmethod
    def get_A_LS_from_trajectory(z_trajectory, t_values):
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

        Z_traj[-1] = ZT_new
        A_lstsq = MatrixODEAutoGradFn.get_A_LS_from_trajectory(z_trajectory=Z_traj, t_values=ctx.t_values)
        ctx.params['A'] = A_lstsq
        return None, None, None, None, None, None


if __name__ == '__main__':
    # Z = mygrad.Tensor([0.1, 0.2])
    # A = mygrad.Tensor([[0.3, 0.4], [0.8, -0.9]])
    # delta_t = 0.01
    # f = mygrad.matmul(A, Z) * delta_t + Z
    # y = f
    # f.backward()
    # print(Z.grad)
    # Z_grad_analytical = A*delta_t + np.identity(2)
    # print(Z_grad_analytical)
    # f_inst = F()
    # z = torch.Tensor([0.2,0.3])
    # z_new = f_inst(z)

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    """
    # Differentiate `loss` with respect to the first positional argument:
    W_grad = grad(loss, argnums=0)(W, b)
    print('W_grad', W_grad)
    
    # Since argnums=0 is the default, this does the same thing:
    W_grad = grad(loss)(W, b)
    print('W_grad', W_grad)
    
    # But we can choose different values too, and drop the keyword:
    b_grad = grad(loss, 1)(W, b)
    print('b_grad', b_grad)
    
    # Including tuple values
    W_grad, b_grad = grad(loss, (0, 1))(W, b)
    print('W_grad', W_grad)
    print('b_grad', b_grad)
    """
    # A = jnp.array([[0.1, 0.2], [0.8, 0.9]])
    # z = jnp.array([0.1, 0.2])
    # delta_t = 0.01
    # j = jacfwd(function1, argnums=0)(z, A, delta_t)
    # print(j)
    # j2 = A*delta_t+jnp.identity(2)
    # print(j2)
    D = 3
    A_ref = torch.distributions.Uniform(-10, 10).sample(sample_shape=torch.Size([3, 3]))
    N = 1024
    batch_size = 64
    t_span = 0, 1
    epochs = 1000
    delta_t = 0.25
    mse_loss = MSELoss()
    ###
    ds = MatrixOdeDataset(D=D, A=A_ref, N=N, t_span=t_span, delta_t=delta_t)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    func = MatrixODEAutoGradFn()
    unif = torch.distributions.Uniform(0.001, 0.005)
    A_train = torch.nn.Parameter(unif.sample(torch.Size([D, D])))
    solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32)
    params = {'A': A_train, 'lr': 2.0}
    alpha = 1.0
    loss_threshold = 1e-3
    for epoch in range(1, epochs + 1):
        batches_losses = []
        mse_loss_fn = MSELoss()
        for i, (Z0, ZT) in enumerate(dl):
            A_old = params['A']
            ZT_hat = func.apply(Z0, params, solver, ode_func, t_span, delta_t)
            # loss = mse_loss(ZT, ZT_hat)
            # Compute and print loss
            # print(f"A norm before = {torch.norm(params['A'])}")
            # loss = (ZT_hat - ZT).pow(2).sum()
            # print(f'loss = {loss.item()}')
            loss = mse_loss_fn(ZT_hat, ZT)  # +10.0*torch.norm(params['A'])
            # print(loss.item())
            loss.backward(retain_graph=True)
            A_new = params['A']
            A_updated = alpha * A_new + (1 - alpha) * A_old
            params['A'] = A_updated
            # print(f"A norm after = {torch.norm(params['A'])}")
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        if epoch % 10 == 0:
            print(f'epoch = {epoch} , loss = {epoch_avg_loss}')
        if epoch_avg_loss <= 1e-3:
            print(f'loss <= threshold = {loss_threshold}, Terminating ! ')
            break
