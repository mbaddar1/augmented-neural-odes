# https://jax.readthedocs.io/en/latest/installation.html
from typing import Any, Callable, Tuple

import mygrad
import numpy as np
import jax.numpy as jnp
import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

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
    dzdt = torch.einsum('ji,bi->bj', A, z)
    return dzdt


class MatrixOdeDataset(Dataset):
    def __init__(self, D, A, N, t_span):
        self.N = N
        unif = torch.distributions.Uniform(-1, 1)
        Z0 = torch.tensor(unif.sample(torch.Size([N, D])),requires_grad=True)
        solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=Z0.dtype)
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
        ctx.params = params
        ctx.z_traj = soln.z_trajectory
        ctx.t_values = soln.t_values
        return ZT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        dz_dt_traj = [grad_outputs[0]]

        return None, None, None, None, None


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
    A = torch.Tensor([[0.1, 0.2, -0.6], [0.8, -0.3, 0.4], [0.1, -0.2, 0.9]])
    N = 1024
    batch_size = 32
    t_span = 0, 1
    epochs = 100
    mse_loss = MSELoss()
    ###
    ds = MatrixOdeDataset(D=D, A=A, N=N, t_span=t_span)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    func = MatrixODEAutoGradFn()
    unif = torch.distributions.Uniform(0.01, 0.05)
    A_train = torch.nn.Parameter(unif.sample(torch.Size([D, D])))
    solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32)
    params = {'A': A_train}
    for i, (Z0, ZT) in enumerate(dl):
        ZT_hat = func.apply(Z0, params,solver, ode_func, t_span)
        # loss = mse_loss(ZT, ZT_hat)
        # Compute and print loss
        loss = (ZT_hat - ZT).pow(2).sum()
        loss.backward()
