# https://mathinsight.org/ordinary_differential_equation_linear_integrating_factor#:~:text=An%20example%20of%20such%20a,easily%20handle%20nonlinearities%20in%20t.
# https://www.cuemath.com/calculus/linear-differential-equation/
# https://en.wikipedia.org/wiki/Recursive_least_squares_filter
# https://www.simiode.org/resources/6425/download/5-010-Text-S-MatrixExponential-StudentVersion.pdf
from typing import Tuple, Callable, Any, List

import numpy as np
import scipy.linalg
import torch
from torch.nn import Sequential, MSELoss
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_ode_solver import TorchODESolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


def ode_func(t: float, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    x : b x D_z
    A : Dz x (Dz+1)
    t : scalar
    """

    assert A.size()[0] == A.size()[1], f"A must be square"
    # z_aug = torch.cat([z, torch.tensor(np.repeat(t, b), dtype=z.dtype).view(b, 1)], dim=1)
    dzdt = torch.einsum('ji,bi->bj', A, z)
    return dzdt


def forward_function(X: torch.Tensor, P: torch.Tensor, A: torch.Tensor, Q: torch.nn.Module, solver: TorchODESolver,
                     ode_func: Callable, t_span: Tuple) -> torch.Tensor:
    # assert P is an Identity Matrix
    assert (len(P.size()) == 2) and (P.size()[0] == P.size()[1]) and (torch.equal(P, torch.eye(P.size()[0])))

    Z0 = torch.einsum('ji,bi->bj', P, X)
    soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
    ZT = soln.z_trajectory[-1]
    y = Q(ZT)
    return y


def forward_function_ode_only(X: torch.Tensor, P: torch.Tensor, A: torch.Tensor, solver: TorchODESolver,
                              ode_func: Callable, t_span: Tuple) -> Tuple[List[torch.Tensor], List[float]]:
    Z0 = torch.einsum('ji,bi->bj', P, X)
    soln = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(A,))
    return soln.z_trajectory, soln.t_values


class MatrixOdeGradientDescentModel(torch.nn.Module):
    def __init__(self, Dx, Dz, Dy, param_ulow, param_uhigh, Q_hidden_dim, ode_solver, ode_func, t_span):
        super().__init__()
        self.t_span = t_span
        self.ode_solver = ode_solver
        self.ode_func = ode_func
        unif = torch.distributions.Uniform(low=param_ulow, high=param_uhigh)
        self.P = torch.eye(Dz)  # FIXME make a param later unif.sample(sample_shape=torch.Size([Dz, Dx]))  # not a param
        self.A = torch.nn.Parameter(unif.sample(sample_shape=torch.Size([Dz, Dz])))
        self.Q = QnnMatrixODE(input_dim=Dz, output_dim=Dy, hidden_dim=Q_hidden_dim, n_layers=Q_nlayers)

    def forward(self, x):
        y = forward_function(X=x, P=self.P, A=self.A, Q=self.Q, solver=solver, ode_func=self.ode_func,
                             t_span=self.t_span)
        return y


class QnnMatrixODE(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, u_low=None, u_high=None):
        super().__init__()
        self.model = Sequential()
        assert n_layers >= 3, "n_layers must >=3"
        internal_layers = n_layers - 2
        self.synthetic = False
        self.model.extend(torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim,dtype=torch.float64), torch.nn.ReLU()))
        for l in range(internal_layers):
            self.model.extend(torch.nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim,dtype=torch.float64), torch.nn.ReLU()))

        self.model.append(torch.nn.Linear(hidden_dim, output_dim,dtype=torch.float64))
        if u_low is not None and u_high is not None:
            for i in range(n_layers):
                layer = self.model[i]
                if isinstance(layer, torch.nn.Linear):
                    layer.weight = torch.nn.Parameter(torch.distributions.Uniform(low=u_low, high=u_high).sample(
                        sample_shape=layer.weight.size()).type(tensor_dtype))
                    layer.bias = torch.nn.Parameter(torch.distributions.Uniform(low=u_low, high=u_high).sample(
                        sample_shape=layer.bias.size()).type(tensor_dtype))
            self.synthetic = True
        else:
            self.synthetic = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if self.synthetic:
            return y.detach()
        else:
            return y


class MatrixODEdataSet(Dataset):

    def __init__(self, N, Dx, P: torch.Tensor, A: torch.Tensor, Q: torch.nn.Module, solver: TorchODESolver,
                 ode_func: Callable, t_span: Tuple, x_ulow: float,
                 x_uhigh: float, tensor_dtype: torch.dtype = torch.float64):
        self.N = N
        X = torch.distributions.Uniform(low=x_ulow, high=x_uhigh).sample(sample_shape=torch.Size([N, Dx])).type(
            tensor_dtype)
        y = forward_function(X=X, P=P, A=A, Q=Q, solver=solver, ode_func=ode_func, t_span=t_span)
        self.x_train = X
        self.y_train = y

    def __len__(self):
        return self.N

    def __getitem__(self, index) -> T_co:
        return self.x_train[index], self.y_train[index]


# https://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
class MatrixOdeTrainableModelLeastSquares(torch.nn.Module):
    def __init__(self, Dx, Dz, Dy, solver, ode_func, t_span, hidden_dim, n_layers):
        super().__init__()
        low_ = 0.01
        high_ = 0.1
        self.P = torch.nn.Parameter(torch.distributions.Uniform(low=low_, high=high_).sample(torch.Size([Dz, Dx])).type(torch.float64))
        self.A = torch.distributions.Uniform(low=low_, high=high_).sample(torch.Size([Dz, Dz])).type(torch.float64)
        self.Q = QnnMatrixODE(input_dim=Dz, output_dim=Dy, hidden_dim=hidden_dim, n_layers=n_layers)
        self.solver = solver
        self.ode_func = ode_func
        self.t_span = t_span

    def forward(self, X):
        Lscust = LsCustomFunc

        zT = Lscust.apply(X, self.P, self.A, self.solver, self.ode_func, self.t_span, self.A)

        y = self.Q(zT)
        return y


class LsCustomFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, X: torch.Tensor, P: torch.Tensor, A: torch.Tensor, solver: TorchODESolver, ode_func: Callable,
                t_span: Tuple, opt_ctx: dict) -> Any:
        z_traj, t_values = forward_function_ode_only(X=X, A=A, P=P, solver=solver, ode_func=ode_func, t_span=t_span)

        ctx.x = X
        ctx.t_values = t_values
        ctx.z_traj = z_traj
        zT = z_traj[-1]
        ctx.A = A
        ctx.P = P
        ctx.solver = solver
        ctx.ode_func = ode_func
        ctx.t_span = t_span
        ctx.opt_ctx = opt_ctx
        return zT

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        alpha = 1.0
        lr = 1e-3
        dl_dzT = grad_outputs[0]
        n_t_values = len(ctx.t_values)
        zT = ctx.z_traj[-1]
        zT_prime = zT - lr * dl_dzT
        E = torch.linalg.lstsq(ctx.x, zT_prime).solution
        logE = scipy.linalg.logm(E.detach())
        A_ls = logE / (ctx.t_span[1] - ctx.t_span[0])
        # sanity check for A_ls
        cust = LsCustomFunc
        zT_prime_hat_1 = cust.apply(ctx.x, ctx.P, ctx.A, ctx.solver, ctx.ode_func, ctx.t_span, ctx.opt_ctx)
        zT_prime_hat_2 = cust.apply(ctx.x, ctx.P, torch.tensor(A_ls, dtype=ctx.x.dtype), ctx.solver, ctx.ode_func,
                                    ctx.t_span, ctx.opt_ctx)
        E2 = torch.tensor(scipy.linalg.expm(ctx.A.detach().numpy() * (ctx.t_span[1] - ctx.t_span[0])))
        zT_prime_hat_3 = torch.einsum("ji,bi->bj", E2, ctx.x)
        err_ = torch.norm(zT_prime_hat_3 - zT_prime_hat_1)  # should be close to zero
        norm1 = torch.norm(zT_prime - zT_prime_hat_1)
        norm2 = torch.norm(zT_prime - zT_prime_hat_2)
        norm_diff = norm2 - norm1  # should be negative
        x = 10
        # batch_size = list(zT.size())[0]
        # z_t_plus_1_prime = zT_prime
        # A_prime = ctx.A  # updated
        # for i in range(n_t_values - 2, -1, -1):
        #     t = ctx.t_values[i]
        #     t_plus_1 = ctx.t_values[i + 1]
        #     delta_t = t_plus_1 - t
        #     zt = ctx.z_traj[i]
        #     delta_z = (z_t_plus_1_prime - zt)
        #     yy = delta_z / delta_t
        #     z_aug = torch.cat([zt, torch.Tensor(np.repeat(t, batch_size)).view(batch_size, 1)], dim=1)
        #     A_ls = torch.linalg.lstsq(z_aug, yy).solution.T
        #     # update
        #     A_prime = alpha * A_ls + (1 - alpha) * A_prime
        #     z_aug_prime = torch.cat(
        #         [z_t_plus_1_prime, torch.Tensor(np.repeat(t_plus_1, batch_size)).view(batch_size, 1)], dim=1)
        #     delta_z_prime = torch.einsum('bi,ji->bj', z_aug_prime, A_prime)
        #     z_t_prime = z_t_plus_1_prime - delta_z_prime * delta_t
        #     z_t_plus_1_prime = z_t_prime
        #     break
        # # TODO checkpoint, check A_ls convergence
        # cst_ = LsCustomFunc
        # zT_prime_hat_1 = cst_.apply(ctx.x, ctx.P, ctx.A, ctx.solver, ctx.ode_func, ctx.t_span, None)
        # zT_prime_hat_2 = cst_.apply(ctx.x, ctx.P, A_prime, ctx.solver, ctx.ode_func, ctx.t_span, None)
        # loss_1 = torch.norm(zT_prime - zT_prime_hat_1)
        # loss_2 = torch.norm(zT_prime - zT_prime_hat_2)
        # delta_loss = loss_2 - loss_1  # must be negative
        # if ctx.opt_ctx is not None:
        #     ctx.opt_ctx['A'] = A_prime
        #
        return None, None, None, None, None, None, None


if __name__ == '__main__':
    Dx = 2
    Dy = 1
    N = 1024
    batch_size = 1
    hidden_dim = 64
    t_span = (0, 0.8)
    x_ulow = -0.5
    x_uhigh = 0.5
    param_synthetic_ulow = 0.5
    param_synthetic_uhigh = 2.1
    param_init_ulow = 0.01
    param_init_uhigh = 0.09
    epochs = 100
    Q_nlayers = 3
    model_type = 'LS'
    tensor_dtype = torch.float64
    #######

    A = 1e-8 * torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=tensor_dtype)
    # eig_vals = torch.linalg.eigvals(A) FIXME , must be sq
    P = torch.tensor([[1.0, 0.0], [0.0, 1.0]],dtype=tensor_dtype)
    Dz = list(P.size())[0]
    synthetic_qnn = QnnMatrixODE(input_dim=Dz, output_dim=Dy, hidden_dim=hidden_dim, u_low=param_synthetic_ulow,
                                 u_high=param_synthetic_uhigh,
                                 n_layers=Q_nlayers)

    solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=tensor_dtype)  # TorchEulerSolver(step_size=0.1)

    ds = MatrixODEdataSet(N=N, Dx=Dx, A=A, P=P, Q=synthetic_qnn, t_span=t_span, x_ulow=x_ulow, x_uhigh=x_uhigh,
                          solver=solver, ode_func=ode_func)

    train_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    del synthetic_qnn
    min_stable_norm = 1e-5
    max_stable_norm = 1e5

    mtx_ode_trainable_grad_desc = MatrixOdeGradientDescentModel(Dx=Dx, Dz=Dz, Dy=Dy, param_ulow=param_init_ulow,
                                                                param_uhigh=param_init_uhigh,
                                                                Q_hidden_dim=hidden_dim, ode_solver=solver,
                                                                ode_func=ode_func,
                                                                t_span=t_span)

    mtx_ode_trainable_LS = MatrixOdeTrainableModelLeastSquares(Dx=Dx, Dz=Dz, Dy=Dy, solver=solver, ode_func=ode_func,
                                                               t_span=t_span, hidden_dim=hidden_dim, n_layers=Q_nlayers)

    if model_type == 'GD':
        model = mtx_ode_trainable_grad_desc
    elif model_type == 'LS':
        model = mtx_ode_trainable_LS
    optimizer = torch.optim.Adam(params=mtx_ode_trainable_grad_desc.parameters(), lr=1e-1)
    loss_fn = MSELoss()
    opt_ctx = {}
    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            assert min_stable_norm < torch.norm(X) < max_stable_norm
            assert min_stable_norm < torch.norm(y) < max_stable_norm

            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            print(loss.item())
            loss.backward()

            optimizer.step()
    print(model.A)

    #####
