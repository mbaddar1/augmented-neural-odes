"""
References

Stability_and_steady_state_of_the_matrix_system
https://en.wikipedia.org/wiki/Matrix_differential_equation#Stability_and_steady_state_of_the_matrix_system
ODE and PDE Stability Analysis
https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture19_pde2.pdf
"""
import random
from typing import Any, Callable
import pandas as pd
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch.nn
from torch.nn import Sequential, MSELoss
from phd_experiments.datasets.toy_ode import ToyODE
from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

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


def basis(x: torch.Tensor, t: float):
    pass


class OdeFuncLinear(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = torch.nn.Linear(in_features=latent_dim, out_features=latent_dim, bias=False)

    def forward(self, t, y):
        dydt = self.net(y ** 3)
        return dydt


class OdeFuncNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(OdeFuncNN, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=hidden_dim, out_features=out_dim),
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
        A = params['A']
        ode_func = params['ode_func']
        solver = TorchEulerSolver(step_size)
        # solver = TorchRK45(device=torch.device('cpu'))
        soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=x)
        ctx.z_traj = soln.z_trajectory
        ctx.t_vals = soln.t_values
        zT = soln.z_trajectory[-1]
        ctx.params = params
        return zT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        lr = 0.01
        zT = ctx.z_traj[-1]
        dLdzT = grad_outputs[0]
        zT_prime = zT - lr * dLdzT
        z_traj_new = ctx.z_traj
        z_traj_new[-1] = zT_prime
        delta_z = pd.Series(data=z_traj_new).diff(1)[1:].values
        delta_t = np.nanmean(pd.Series(data=ctx.t_vals).diff(1)[1:].values)  # assume fixed delta-t
        X_ls = torch.concat(tensors=ctx.z_traj[:-1])
        Y_ls = torch.concat(tensors=list(delta_z / delta_t))
        ls_soln = torch.linalg.lstsq(X_ls, Y_ls)
        A_ls = ls_soln.solution
        ctx.params['A'] = A_ls
        return None, None


class HybridMatrixNeuralODE(torch.nn.Module):
    def __init__(self, latent_dim, nn_hidden_dim, out_dim, opt_method: str, ode_func: Callable):
        super().__init__()
        self.opt_method = opt_method
        self.ode_func = ode_func
        self.Q = Sequential(torch.nn.Linear(latent_dim, nn_hidden_dim), torch.nn.Tanh(),
                            torch.nn.Linear(nn_hidden_dim, out_dim))
        # Sequential(
        # torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        # torch.nn.Identity(),
        # torch.nn.Linear(in_features=hidden_dim, out_features=out_dim))
        unif_low = 0.001
        unif_high = 0.005
        A_init = torch.distributions.Uniform(unif_low, unif_high).sample(
            sample_shape=torch.Size([(latent_dim + 1), latent_dim]))
        if opt_method == 'lstsq':
            self.A = A_init
            self.params = {}
        elif opt_method == 'graddesc':
            self.A = torch.nn.Parameter(A_init)
        else:
            raise ValueError(f'Unknown opt-method = {opt_method}')

    def forward(self, x):
        t_span = 0, 1
        step_size = 0.02
        solver = TorchEulerSolver(step_size=step_size)
        if self.opt_method == 'lstsq':
            fn = EulerFunc()
            self.params = {'A': self.A, 't_span': t_span, 'ode_func': self.ode_func, 'step_size': step_size,
                           'solver': solver}
            zT = fn.apply(x, self.params)
        elif self.opt_method == 'graddesc':
            soln = solver.solve_ivp(func=self.ode_func, t_span=t_span, z0=x)
            zT = soln.z_trajectory[-1]
        else:
            raise ValueError(f'Unknown opt method = {self.opt_method}')
        y_hat = self.Q(zT)
        return y_hat


if __name__ == '__main__':
    epochs = 10000
    batch_size = 128
    lr = 1e-3
    N = 1000

    # Fixme add normalization
    # https://inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/
    # overall_dataset = ToyRelu(input_dim=input_dim, out_dim=1, N=N)
    overall_dataset = ToyODE()
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    latent_dim = input_dim
    nn_hidden_dim = 50

    ##
    splits = random_split(dataset=overall_dataset, lengths=[0.8, 0.2])
    train_dataset = splits[0]
    test_dataset = splits[1]
    opt_method = 'graddesc'
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    ode_func_nn_instance = OdeFuncNN(input_dim=input_dim, hidden_dim=nn_hidden_dim, out_dim=output_dim)
    ode_func_linear_instance = OdeFuncLinear(latent_dim=latent_dim)
    model = HybridMatrixNeuralODE(latent_dim=latent_dim, nn_hidden_dim=nn_hidden_dim, out_dim=output_dim,
                                  opt_method=opt_method,
                                  ode_func=ode_func_linear_instance)
    optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    loss_fn = MSELoss()
    epochs_avg_losses = []
    print(opt_method)
    alpha = 0.8
    for epoch in range(epochs):
        batches_losses = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            if opt_method == 'lstsq':
                X = torch.nn.Parameter(X)  # FIXME : a hack to make backward in autograd.func be fired !!
            y_hat = model(X)
            if opt_method == 'lstsq':
                A_old = model.params['A']
            loss = loss_fn(y, y_hat)
            # print(f'loss = {loss.item()}')
            # print(f'A = {ode_func_linear_instance.net.weight}')
            batches_losses.append(loss.item())
            loss.backward()
            if opt_method == 'lstsq':
                A_ls = model.params['A']
                e = torch.norm(A_ls - A_old)
                model.A = alpha * A_ls + (1 - alpha) * A_old

            optimizer.step()
        epochs_avg_losses.append(np.nanmean(batches_losses))
        if epoch % 10 == 0:
            print(f'epoch {epoch} loss = {epochs_avg_losses[-1]}')
