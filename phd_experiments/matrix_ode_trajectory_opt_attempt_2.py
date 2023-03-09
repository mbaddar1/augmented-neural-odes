# https://jax.readthedocs.io/en/latest/installation.html
from typing import Any, Callable, Tuple

import mygrad
import numpy as np
import jax.numpy as jnp
import pandas as pd
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
        #
        # Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        # solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=Z0.dtype)
        # soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        # ZT = soln.z_trajectory[-

        # one euler step
        unif = torch.distributions.Uniform(-1, 1)
        delta_t = t_span[1] - t_span[0]
        Z0 = torch.tensor(unif.sample(torch.Size([N, D])), requires_grad=True)
        ZT = torch.einsum('ji,bi->bj', A, Z0) * delta_t + Z0
        self.x_train = Z0
        self.y_train = ZT

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def __len__(self):
        return self.N


class MatrixODEAutoGradFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, Z0: torch.Tensor, params: dict, solver: TorchRK45, ode_func: Callable, t_span: Tuple) -> Any:
        # A = params['A']
        # soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
        # ZT = soln.z_trajectory[-1]
        # ctx.params = params
        # ctx.z_traj = soln.z_trajectory
        # ctx.t_values = soln.t_values
        # ctx.solver = solver
        # ctx.t_span = t_span
        # ctx.ode_func = ode_func
        A = params['A']
        delta_t = t_span[1] - t_span[0]
        ZT = torch.einsum('ji,bi->bj', A, Z0) * delta_t + Z0
        ctx.ZT = ZT
        ctx.t_span = t_span
        ctx.Z0 = Z0
        ctx.params = params
        return ZT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # dz_dt_traj = [grad_outputs[0]]
        dLdz_T = grad_outputs[0]
        # FIXME, invest some-time modifying it
        lr = 0.1
        ZT = ctx.ZT
        ZT_new = ZT - lr * dLdz_T
        delta_t = ctx.t_span[1] - ctx.t_span[0]
        Y_lstsq = (ZT_new - ctx.Z0) / delta_t
        X_lstsq = ctx.Z0
        lstsq_sol = torch.linalg.lstsq(X_lstsq, Y_lstsq)
        A_lstsq = lstsq_sol.solution.T
        ctx.params['A'] = A_lstsq

        # dummy = lr *dLdz_T
        # A = ctx.params['A']
        # I = torch.eye(A.size()[0])
        #
        # ZT_new = ZT - lr * dLdz_T
        # z_trajectory_new = [ZT_new]  # ZT_new = ZT-0.1*dL/dZT
        # T = len(ctx.t_values)
        # dz_Tdz_t_mul_acc = torch.eye(A.size()[0])
        # for t_idx in range(T - 2, 0, -1):
        #     t = ctx.t_values[t_idx]
        #     t_plus_1 = ctx.t_values[t_idx + 1]
        #     delta_t = t_plus_1 - t
        #     dz_t_plus_1_dz_t = A * delta_t + I
        #     dz_Tdz_t = torch.matmul(dz_Tdz_t_mul_acc, dz_t_plus_1_dz_t)
        #     dL_dz_t = torch.einsum('bi,ij->bj', dLdz_T, dz_Tdz_t)
        #     zt_new = ctx.z_traj[t_idx] - lr * dL_dz_t
        #     # updates
        #     dz_Tdz_t_mul_acc = torch.matmul(dz_Tdz_t_mul_acc,dz_t_plus_1_dz_t)
        #     z_trajectory_new.insert(0, zt_new)
        # z_trajectory_new.insert(0, ctx.z_traj[0])
        #
        # delta_z = list(pd.Series(z_trajectory_new).diff(1).values[1:])
        # delta_t = list(pd.Series(ctx.t_values).diff(1).values[1:])
        # delta_z_delta_t_zip = list(zip(delta_z, delta_t))
        # Y_lstsq = torch.concat(list(map(lambda x: x[0] / x[1], delta_z_delta_t_zip)), dim=0)
        # X_lstsq = torch.concat(z_trajectory_new[:-1], dim=0)
        # lstsq_soln = torch.linalg.lstsq(X_lstsq, Y_lstsq)
        # A_new = lstsq_soln.solution.T
        #
        # # sanity check
        # # Z0_new = z_trajectory_new[0]
        # # ZT_new = z_trajectory_new[-1]
        # # #soln_new = ctx.solver.solve_ivp(func=ctx.ode_func, t_span=ctx.t_span, z0=Z0_new, args=(A_new,))
        # # ZT_hat_new = soln_new.z_trajectory[-1]
        # # mse_loss = MSELoss()
        # # err = mse_loss(ZT_new, ZT_hat_new)
        #
        # ctx.params['A'] = A_new

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
    A_ref = torch.distributions.Uniform(-10, 10).sample(sample_shape=torch.Size([3, 3]))
    N = 1024
    batch_size = 64
    t_span = 0, 1
    epochs = 1000
    mse_loss = MSELoss()
    ###
    ds = MatrixOdeDataset(D=D, A=A_ref, N=N, t_span=t_span)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    func = MatrixODEAutoGradFn()
    unif = torch.distributions.Uniform(0.001, 0.005)
    A_train = torch.nn.Parameter(unif.sample(torch.Size([D, D])))
    solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float32)
    params = {'A': A_train}
    alpha = 1.0

    for epoch in range(1, epochs + 1):
        batches_losses = []
        mse_loss_fn = MSELoss()
        for i, (Z0, ZT) in enumerate(dl):
            A_old = params['A']
            ZT_hat = func.apply(Z0, params, solver, ode_func, t_span)
            # loss = mse_loss(ZT, ZT_hat)
            # Compute and print loss
            # print(f"A norm before = {torch.norm(params['A'])}")
            # loss = (ZT_hat - ZT).pow(2).sum()
            # print(f'loss = {loss.item()}')
            loss = mse_loss_fn(ZT_hat, ZT)  # +10.0*torch.norm(params['A'])
            print(loss.item())
            loss.backward(retain_graph=True)
            A_new = params['A']
            A_updated = alpha * A_new + (1 - alpha) * A_old
            params['A'] = A_updated
            # print(f"A norm after = {torch.norm(params['A'])}")
            batches_losses.append(loss.item())
        epoch_avg_loss = np.nanmean(batches_losses)
        if epoch % 10 == 0:
            print(f'epoch = {epoch} , loss = {epoch_avg_loss}')
