from typing import Any, Callable, Tuple, Iterable, List

import pandas as pd
import torch.autograd
from torch import Tensor

from dlra.feature_utils import PolyBasis
from dlra.linearbackend import lb
from dlra.tt import TensorTrain, Extended_TensorTrain
from phd_experiments.tn.tt import TensorTrainFixedRank
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

"""
https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html 

"""


class Forward2():
    @staticmethod
    def forward2(x: torch.Tensor, P: torch.Tensor, input_dimensions: Iterable[int],
                 W: [TensorTrainFixedRank | List[TensorTrain]], tensor_dtype: torch.dtype,
                 tt_ode_func: Callable, t_span: Tuple, basis_fn: str, basis_params: dict) \
            -> Tuple[Iterable[Tensor], Iterable[float]]:
        z0_contract_dims = list(range(1, len(input_dimensions) + 1))
        P_contract_dims = list(range(len(input_dimensions)))
        z0 = torch.tensordot(a=x, b=P, dims=(z0_contract_dims, P_contract_dims))
        torch_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=tensor_dtype, is_batch=True)
        soln = torch_solver.solve_ivp(func=tt_ode_func, t_span=t_span, z0=z0,
                                      args=(W, basis_fn, basis_params))

        return soln.z_trajectory, soln.t_values


class TensorTrainContainer:
    def __init__(self):
        self.tt = None


class TTOdeAls(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, P: torch.Tensor, input_dimensions: Iterable[int],
                W: [TensorTrainFixedRank | List[TensorTrain]], tt_container: TensorTrainContainer,
                tensor_dtype: torch.dtype,
                tt_ode_func: Callable, t_span: Tuple, basis_fn: str, basis_params: dict) -> torch.Tensor:
        ctx.z_trajectory, ctx.t_values = Forward2.forward2(x, P, input_dimensions, W, tensor_dtype, tt_ode_func, t_span,
                                                           basis_fn,
                                                           basis_params)
        ctx.W = W
        ctx.P = P
        ctx.x = x
        return ctx.z_trajectory[-1]

    @staticmethod
    def backward(ctx: Any, grad_outputs: Tensor) -> Any:
        alpha = 0.1
        lr = 0.001
        eps = 1e-8
        zT = ctx.z_trajectory[-1]
        dL_dzT = grad_outputs
        z_t_plus_1_prime = zT - lr * dL_dzT
        trajectory_len = len(ctx.z_trajectory)
        poly_deg = 3  # fixme make as param
        batch_size = list(zT.size())[0]
        Dz = list(zT.size())[1]
        poly_basis = PolyBasis(degrees=[poly_deg + 1] * (Dz + 1))
        # poly_deg+1 => dim as poly raises to power 0 , 1, ..., degree
        # Dz+1 , +1 for time augmentation
        W_last = ctx.W
        for i in range(trajectory_len - 2, -1, -1):
            t = ctx.t_values[i]
            t_plus_1 = ctx.t_values[i + 1]
            delta_t = t_plus_1 - t
            assert delta_t > 1e-8, f"delta_t is too small : {delta_t} < {eps}"
            yy = (z_t_plus_1_prime - ctx.z_trajectory[i]) / delta_t
            xx = ctx.z_trajectory[i]
            t_tensor = torch.Tensor([t] * batch_size).view(batch_size, 1)
            # concat time
            xx = torch.concat([xx, t_tensor], dim=1)
            W_yy_pair = pd.DataFrame({'W': W_last, 'yy': list(yy.T)})
            # apply als
            W_als = W_yy_pair.apply(lambda v: TTOdeAls._als(v[0], v[1].view(batch_size, -1), xx, poly_basis),
                                    axis=1).values
            # update W and Z
            W_last = [alpha * W_als[i] + (1 - alpha) * W_last[i] for i in range(Dz)]
            z_t_plus_1_prime = torch.concat(tensors=list(
                map(lambda w: TTOdeAls._apply_dzdt_fn(w, xx, poly_basis), W_last)), dim=1)  # TODO or W_als ??
        # FIXME , for debug only, and apply norm of diffs not diffs of norms
        W_norm_diff = sum([W_last[i].norm() - ctx.W[i].norm() for i in range(len(ctx.W))])
        ret = torch.linalg.lstsq(ctx.x, z_t_plus_1_prime)
        P_als = ret.solution
        P_prime = alpha * P_als + (1 - alpha) * ctx.P  # TODO get P_als or smoothen with alpha ?
        diff_ = torch.norm(P_als - ctx.P)
        diff2_ = torch.norm(P_prime - ctx.P)
        return None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _als(W: TensorTrain, yy_vec: torch.Tensor, xx_tensor: torch.Tensor, basis_fn: Any) -> TensorTrain:
        # TODO understand the rank change due to add fun in TT
        # infer ranks from cores
        n_comps = len(W.comps)
        ranks = [list(comp.size())[2] for comp in W.comps[:n_comps - 1]]
        xTT = Extended_TensorTrain(tfeatures=basis_fn, ranks=ranks, comps=W.comps)
        norm_before = xTT.tt.norm()
        xTT.fit(x=lb.tensor(xx_tensor), y=lb.tensor(yy_vec), rule=None, verboselevel=1, reg_param=1e-6, iterations=2000)
        norm_after = xTT.tt.norm()
        diff_ = norm_after - norm_before
        return xTT.tt

    @staticmethod
    def _apply_dzdt_fn(W: TensorTrain, xx: [torch.Tensor, lb.tensor], basis_fn: Any) -> torch.Tensor:
        n_comps = len(W.comps)
        ranks = [list(comp.size())[2] for comp in W.comps[:n_comps - 1]]
        xTT = Extended_TensorTrain(tfeatures=basis_fn, ranks=ranks, comps=W.comps)
        z = xTT(xx)
        return z
