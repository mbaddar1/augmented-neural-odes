"""
Bad Gradient in pytorch
https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
https://discuss.pytorch.org/t/how-to-check-for-vanishing-exploding-gradients/9019/3

torch einsum autograd
https://discuss.pytorch.org/t/automatic-differentation-for-pytorch-einsum/112504

PyTorch Reproducibility
https://pytorch.org/docs/stable/notes/randomness.html

SEED value selection
https://www.linkedin.com/pulse/how-choose-seed-generating-random-numbers-rick-wicklin/
https://blogs.sas.com/content/iml/2017/06/01/choose-seed-random-number.html
Typical SEED Values
42
large primes https://bigprimes.org/
18819191
71623183
71623183
from here https://www.linkedin.com/pulse/how-choose-seed-generating-random-numbers-rick-wicklin/
12345
54321
987654321

adjustable (adaptive) learning rate (using pytorch optim schedulers)
https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no

oscillating loss
https://ai.stackexchange.com/questions/14079/what-could-an-oscillating-training-loss-curve-represent

Learning rate adjustment
https://www.jeremyjordan.me/nn-learning-rate/

getting nan loss with SGD
https://stackoverflow.com/a/37242531

Data Standardization and Normalization
# X_max = torch.max(X)
            # X_min = torch.min(X)
            # FIXME , for now no normalization is applied, do we need it ?
            #   1. https://www.baeldung.com/cs/normalizing-inputs-artificial-neural-network
            #   2. https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network
            #   3. https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
            #   ==> seems we need to scale INPUT and OUTPUT ( BOTH)
            if scaler is None:
                X_tensor_norm = X
            else:

                X_np = X.detach().numpy()
                # # fixme , debug code
                # min_X_np, max_X_np, avg_X_np = \
                #     np.amin(X_np), np.amax(X_np), np.mean(X_np)
                scaler.fit(X)
            # scaler.fit(X_np)
            # X_norm_np = scaler.transform(X_np)
            # min_X_np_norm, max_X_np_norm, avg_X_np_norm = \
            #     np.amin(X_norm_np), np.amax(X_norm_np), np.mean(X_norm_np)
            # X_tensor_norm = torch.tensor(X_norm_np)
            # fixme , skip normalization
            X_tensor_norm = X
            # fixme, end debug code
            # torch.nn.Sigmoid()(X)  # (X - X_min) / (X_max - X_min)
Batch-Norm refs:
https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
https://arxiv.org/abs/1502.03167
https://medium.com/dejunhuang/learning-day-20-batch-normalization-concept-and-usage-in-pytorch-a8d077d16533

"""

import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
from typing import List
from datetime import datetime

import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from phd_experiments.hybrid_node_tt.torch_rbf import RBF, basis_func_dict

torch.use_deterministic_algorithms(True)
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
from tqdm import tqdm
import random
from phd_experiments.hybrid_node_tt.models import TensorTrainFixedRank


class VDP(Dataset):
    # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    # https://www.johndcook.com/blog/2019/12/22/van-der-pol/f
    # https://arxiv.org/pdf/0803.1658.pdf
    # todo add plotting
    def __init__(self, mio: float, N: int, x_gen_norm_mean: float,
                 x_gen_norm_std: float, normalize_X: bool,
                 normalize_Y: bool, train_or_test: str):
        self.train_or_test = train_or_test
        self.N = N
        self.x_gen_norm_mean = x_gen_norm_mean
        self.x_gen_norm_std = x_gen_norm_std
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y
        Dx_vdp = 2
        self.mio = mio
        assert train_or_test in ["train", "test"]
        self.X = torch.distributions. \
            Normal(loc=self.x_gen_norm_mean, scale=self.x_gen_norm_std). \
            sample(torch.Size([self.N, Dx_vdp]))
        x1 = self.X[:, 0].view(-1, 1)
        x2 = self.X[:, 1].view(-1, 1)
        x1_dot = x2
        x2_dot = mio * (1 - x1 ** 2) * x2 - x1
        self.Y = torch.cat([x1_dot, x2_dot], dim=1)
        # for target std
        self.Y_mean = torch.mean(self.Y, dim=0)
        self.Y_std = torch.std(self.Y, dim=0)
        if normalize_Y:
            self.Y = (self.Y - self.Y_mean) / self.Y_std
        # normalize data from the source
        if normalize_X:
            X_mean = torch.mean(self.X, dim=0)
            X_std = torch.std(self.X, dim=0)
            self.X = (self.X - X_mean) / X_std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_Y_mean(self):
        return self.Y_mean

    def get_Y_std(self):
        return self.Y_std

    def __str__(self):
        return f"***\n" \
               f"VDP Dataset\n" \
               f"N={self.N}\n" \
               f"mio = {self.mio}\n" \
               f"x_gen_norm_mean = {self.x_gen_norm_mean}\n" \
               f"x_gen_norm_std = {self.x_gen_norm_std}\n" \
               f"normalize_X = {self.normalize_X}\n" \
               f"normalize_Y = {self.normalize_Y}\n" \
               f"train-or-test = {self.train_or_test}\n" \
               f"***"


class TrueLambda(torch.nn.Module):
    def __init__(self, A_true):
        super().__init__()
        self.A_true = A_true

    def forward(self, y):
        return torch.mm(y ** 3, self.A_true)


# class YeqAmultYpow3(Dataset):
#     # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L34
#     def __init__(self, A_true,x_gen_norm,):
#         self.true_lambda = TrueLambda(A_true=A_true)
#
#     pass


class LorenzSystem(Dataset):
    # High Dim non-linear systems
    # https://tglab.princeton.edu/wp-content/uploads/2011/03/Mol410Lecture13.pdf (P 2)
    # https://en.wikipedia.org/wiki/Lorenz_system
    def __init__(self, N: int, rho: float, sigma: float, beta: float,
                 x_gen_norm_mean: float, x_gen_norm_std: float,
                 normalize_X: bool, normalize_Y: bool,
                 train_or_test: str):
        self.train_or_test = train_or_test
        self.N = N
        Dx = 3
        # from wikipedia picture
        # https://en.wikipedia.org/wiki/Lorenz_system
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.normalize_X = normalize_X
        self.normalize_Y = normalize_Y
        self.X = torch.distributions.Normal(x_gen_norm_mean, x_gen_norm_std). \
            sample(torch.Size([self.N, Dx]))
        x1 = self.X[:, 0]
        x2 = self.X[:, 1]
        x3 = self.X[:, 2]
        dx1dt = sigma * (x2 - x1).view(-1, 1)
        dx2dt = (x1 * (rho - x3) * x3).view(-1, 1)
        dx3dt = (x1 * x2 - beta * x3).view(-1, 1)
        self.Y = torch.cat([dx1dt, dx2dt, dx3dt], dim=1)
        # normalize or not
        self.Y_mean = torch.mean(self.Y, dim=0)
        self.Y_std = torch.std(self.Y, dim=0)
        X_mean = torch.mean(self.X, dim=0)
        X_std = torch.std(self.X, dim=0)
        if self.normalize_X:
            self.X = (self.X - X_mean) / X_std
        if self.normalize_Y:
            self.Y = (self.Y - self.Y_mean) / self.Y_std

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def get_Y_mean(self):
        return self.Y_mean

    def get_Y_std(self):
        return self.Y_std

    def __str__(self):
        return "\n***\n" \
               "Lorenz-System\n" \
               f"N = {self.N}" \
               f"rho = {self.rho}\n" \
               f"sigma = {self.sigma}\n" \
               f"beta = {self.beta}\n" \
               f"normalize_X = {self.normalize_X}" \
               f"normalize_Y = {self.normalize_Y}" \
               f"****\n"


class FVDP(Dataset):
    # http://math.colgate.edu/~wweckesser/pubs/FVDPI.pdf
    pass


class Y_eq_a_sinX_cosX_dataset(Dataset):
    def __init__(self, input_dim, N):
        self.N = N
        W_dict = {1: torch.tensor([0.1]).view(1, 1),
                  2: torch.tensor([0.1, -0.2]).view(1, 2),
                  4: torch.tensor([0.1, -0.2, 0.3, -0.8]).view(1, 4)}
        W = W_dict[input_dim]
        self.X = torch.distributions.Normal(0, 1).sample(torch.Size([N, input_dim]))
        self.y = torch.einsum('ij,bj->b', W, 0.5 * torch.sin(self.X) + 0.5 * torch.cos(self.X)).view(-1, 1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


####### Basis Funs #############
def get_poly_basis_list(X, deg):
    poly_basis_list = []
    b = X.size()[0]
    dim = X.size()[1]
    for dim_idx in range(dim):
        X_d = X[:, dim_idx]
        x_list = []
        for deg_idx in range(deg + 1):
            x_list.append(torch.pow(X_d, deg_idx).view(-1, 1))
        poly_tensor_per_dim = torch.cat(x_list, dim=1)
        poly_basis_list.append(poly_tensor_per_dim)
    return poly_basis_list


########## Models ##############


class RBFN(torch.nn.Module):
    # https://en.wikipedia.org/wiki/Radial_basis_function_network
    def __init__(self, in_dim, out_dim, n_centres, basis_fn_str, input_batch_norm):
        super().__init__()
        self.basis_fn_str = basis_fn_str
        self.n_centres = n_centres
        self.out_dim = out_dim
        self.in_dim = in_dim
        basis_fn = basis_func_dict()[basis_fn_str]
        self.rbf_module = RBF(in_features=in_dim, n_centres=n_centres, basis_func=basis_fn)
        # rbf inited by its own reset fn
        self.input_norm_module = \
            torch.nn.BatchNorm1d(num_features=in_dim, affine=True) if input_batch_norm \
                else torch.nn.Identity()
        self.linear_module = torch.nn.Linear(in_features=n_centres, out_features=out_dim)
        # TODO revisit theory for batch-norm
        #   ref : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        #   ref : https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #   ref-paper : https://arxiv.org/abs/1502.03167
        #   Note : batch-norm layer is before the non-linearity
        self.net = torch.nn.Sequential(self.input_norm_module,
                                       self.rbf_module, self.linear_module)
        # init
        torch.nn.init.normal_(self.linear_module.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.linear_module.bias, val=0)
        # get numel learnable
        self.numel_learnable = 0
        param_list = list(self.named_parameters())
        for name, param in param_list:
            self.numel_learnable += torch.numel(param)

    def forward(self, X):
        # fixme, several steps for debugging
        mean1 = torch.mean(X, dim=0)
        # std1 = torch.std(X, dim=0)
        # batchnorm1d_out = self.batch_norm_1d_module(X)
        # mean2 = torch.mean(batchnorm1d_out, dim=0)
        # std2 = torch.std(batchnorm1d_out, dim=0)
        # rbfn_out = self.rbf_module(batchnorm1d_out)
        # y_hat = self.linear_module(rbfn_out)
        # fixme : end debugging code
        y_hat = self.net(X)
        return y_hat

    def __str__(self):
        return f"\n***\n" \
               f"RBFN\n" \
               f"in_dim={self.in_dim}\nn_centres={self.n_centres}\n" \
               f"out_dim={self.out_dim}\nbasis_fn={self.basis_fn_str}\n" \
               f"numel_learnable={self.numel_learnable}" \
               f"\n***\n"


class TTRBF(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, order: int,
                 num_rbf_centers: int, rbf_basis_function_name: str,
                 tt_rank: int):
        super().__init__()
        self.num_rbf_centers = num_rbf_centers
        self.order = order
        self.tt_rank = tt_rank
        init_param_norm_mean = 0
        init_param_norm_std = 0.1
        assert order == 4  # fixme, generalize later
        assert rbf_n_centres % order == 0
        self.dim = int(rbf_n_centres / order)
        self.rbf_module = RBF(in_features=in_dim, n_centres=num_rbf_centers,
                              basis_func=basis_func_dict()[rbf_basis_function_name])
        self.G0 = torch.nn.Parameter(torch.empty(self.dim, tt_rank))
        torch.nn.init.normal_(self.G0, mean=init_param_norm_mean, std=init_param_norm_std)

        self.G1 = torch.nn.Parameter(torch.empty(tt_rank, self.dim, tt_rank))
        torch.nn.init.normal_(self.G1, mean=init_param_norm_mean, std=init_param_norm_std)

        self.G2 = torch.nn.Parameter(torch.empty(tt_rank, self.dim, tt_rank))
        torch.nn.init.normal_(self.G2, mean=init_param_norm_mean, std=init_param_norm_std)

        self.G3 = torch.nn.Parameter(torch.empty(tt_rank, self.dim, out_dim))
        torch.nn.init.normal_(self.G3, mean=init_param_norm_mean, std=init_param_norm_std)

        # get num learnable numel
        self.learnable_numel = 0
        for name, param in self.named_parameters():
            self.learnable_numel += torch.numel(param)

    def forward(self, X):
        rbf_out = self.rbf_module(X)
        Phi0 = rbf_out[:, 0:self.dim]
        Phi1 = rbf_out[:, self.dim:2 * self.dim]
        Phi2 = rbf_out[:, 2 * self.dim:3 * self.dim]
        Phi3 = rbf_out[:, 3 * self.dim:4 * self.dim]

        R0 = torch.einsum('di,bd->bi', self.G0, Phi0)
        R1 = torch.einsum('idj,bd->ibj', self.G1, Phi1)
        R2 = torch.einsum('idj,bd->ibj', self.G2, Phi2)
        R3 = torch.einsum('jdl,bd->jbl', self.G3, Phi3)
        res = torch.einsum('bi,ibj,jbk,kbl->bl', R0, R1, R2, R3)
        return res

    def __str__(self):
        return "***\n" \
               "TTRBF\n" \
               f"order = {self.order}\n" \
               f"num_rbf_centers= {self.num_rbf_centers}\n" \
               f"tt_rank = {self.tt_rank}\n" \
               f"dim = {self.dim}\n" \
               f"learnable_numel = {self.learnable_numel}\n" \
               f"***\n"


class FullTensorPoly4dim(torch.nn.Module):
    def __init__(self, input_dim, out_dim, deg):
        super().__init__()
        u_low = -0.05
        u_high = 0.05
        self.deg = deg
        assert input_dim == 4
        self.W = torch.nn.Parameter(
            torch.distributions.Uniform(u_low, u_high).sample(torch.Size(input_dim * [deg + 1])))

    def forward(self, X):
        poly_basis_list = get_poly_basis_list(X, self.deg)
        einsum_params = [self.W] + poly_basis_list
        einsum_str = "acde,ba,bc,bd,be->b"
        res = torch.einsum(einsum_str, einsum_params).view(-1, 1)
        return res


class TTpoly1dim(torch.nn.Module):
    def __init__(self, in_dim, out_dim, deg):  # no rank
        super().__init__()
        assert in_dim == 1
        self.deg = deg
        u_low = -0.05
        u_high = 0.05
        self.G0 = torch.nn.Parameter(torch.distributions.Uniform(u_low, u_high).sample(torch.Size([deg + 1])))

    def forward(self, X):
        poly_tensor_list = get_poly_basis_list(X, self.deg)
        res = torch.einsum("a,ba->b", self.G0, poly_tensor_list[0]).view(-1, 1)
        return res


class TTpoly2dim(torch.nn.Module):
    def __init__(self, in_dim, out_dim, deg, rank):
        super().__init__()
        assert in_dim == 2
        self.deg = deg
        u_low = -0.05
        u_high = 0.05
        self.G0 = torch.nn.Parameter(torch.distributions.Uniform(u_low, u_high).sample(torch.Size([deg + 1, rank])))
        self.G1 = torch.nn.Parameter(torch.distributions.Uniform(u_low, u_high).sample(torch.Size([rank, deg + 1])))

    def forward(self, X):
        poly_tensor_list = get_poly_basis_list(X, self.deg)
        res = torch.einsum("ac,ba,cd,bd->b", self.G0, poly_tensor_list[0], self.G1, poly_tensor_list[1]).view(-1, 1)
        return res


class TTpoly4dim(torch.nn.Module):
    # https://soham.dev/posts/polynomial-regression-pytorch/
    # https://vamsibrp.github.io/pytorch-learning-tutorial/
    def __init__(self, in_dim, out_dim, rank, deg):
        super().__init__()
        # fixme, specific test case
        self.deg = deg
        self.rank = rank
        assert in_dim == 4
        self.order = in_dim
        self.G0 = torch.nn.Parameter(torch.empty(deg + 1, rank))
        self.G1 = torch.nn.Parameter(torch.empty(rank, deg + 1, rank))
        self.G2 = torch.nn.Parameter(torch.empty(rank, deg + 1, rank))
        self.G3 = torch.nn.Parameter(torch.empty(rank, deg + 1))
        torch.nn.init.xavier_normal_(self.G0)
        torch.nn.init.xavier_normal_(self.G1)
        torch.nn.init.xavier_normal_(self.G2)
        torch.nn.init.xavier_normal_(self.G3)
        ####

    def forward(self, X):
        # generate Phi
        poly_basis_list = get_poly_basis_list(X, self.deg)
        einsum_params = [self.G0, poly_basis_list[0], self.G1, poly_basis_list[1], self.G2, poly_basis_list[2], self.G3,
                         poly_basis_list[3]]
        einsum_str = "ac,ba,cfe,bf,ehi,bh,iq,bq->b"
        res = torch.einsum(einsum_str, einsum_params)
        return res.view(-1, 1)


class TTpoly2in2out(torch.nn.Module):
    """
    mainly for VDP traj modeling
    """

    def __init__(self, rank, deg):
        super().__init__()
        out_dim = 2
        self.rank = rank
        self.deg = deg
        self.G0 = torch.nn.Parameter(torch.empty(deg + 1, rank))
        self.G1 = torch.nn.Parameter(torch.empty(rank, deg + 1, out_dim))
        torch.nn.init.normal_(self.G0, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.G1, mean=0.0, std=1.0)

        # get numel learnable
        named_params_list = self.named_parameters()
        self.numel_learnable = 0

        for name, param in named_params_list:
            self.numel_learnable += torch.numel(param)

    def forward(self, X):
        poly_basis_list = get_poly_basis_list(X, self.deg)
        R0 = torch.einsum('dr,bd->br', self.G0, poly_basis_list[0])
        R1 = torch.einsum('rdl,bd->brl', self.G1, poly_basis_list[1])
        res = torch.einsum('br,brl->bl', R0, R1)
        return res

    def __str__(self):
        return f"***\nTT-poly\nrank={self.rank}\ndeg={self.deg}\n" \
               f"numel_learnable = {self.numel_learnable}\n***"


class LinearModeEinSum(torch.nn.Module):
    # TODO
    #   1. a linear model implemented by torch.nn.Param and einsum instead of linear apply
    #   2. make dims and num of el the same as the vanilla model
    #   3. compare grad vals and compu graph.
    def __init__(self, in_dim, out_dim):
        """
        einsum impl. for AX+b
        A,b will be torch.nn.Parameters
        AX will be impl. via torch.nn.sum
        """
        super().__init__()
        # https://stackoverflow.com/questions/64507404/defining-named-parameters-for-a-customized-nn-module-in-pytorch
        u_lower = -0.01
        u_upper = 0.01
        self.A = torch.nn.Parameter(
            torch.nn.Parameter(torch.distributions.Uniform(u_lower, u_upper).sample(torch.Size([in_dim, out_dim]))))
        self.b = torch.nn.Parameter(
            torch.nn.Parameter(torch.distributions.Uniform(u_lower, u_upper).sample(torch.Size([1, out_dim]))))

    def forward(self, X):
        term = torch.einsum("bi,ij->bj", X, self.A)
        y_hat = term + self.b
        return y_hat


class PolyLinearEinsum(LinearModeEinSum):
    def __init__(self, in_dim, out_dim, deg):
        # fixme, specific test-case
        assert in_dim == 4
        super().__init__(in_dim * (deg + 1), out_dim)
        self.deg = deg

    def forward(self, X):
        x_pow_list = []
        for d in range(self.deg + 1):
            x_pow_list.append(torch.pow(X, d))
        X_pow_aug = torch.cat(x_pow_list, dim=1)
        term = torch.einsum("bi,ij->bj", X_pow_aug, self.A)
        y_hat = term + self.b
        return y_hat


class NNmodel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_batch_norm):
        super(NNmodel, self).__init__()

        # TODO revisit theory for batch-norm
        #   ref : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        #   ref : https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #   ref-paper : https://arxiv.org/abs/1502.03167
        #   Note : batch-norm layer is before the non-linearity
        input_norm_module = torch.nn.BatchNorm1d(hidden_dim, affine=True) if input_batch_norm else torch.nn.Identity()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                       input_norm_module,
                                       torch.nn.Tanh(),
                                       torch.nn.Linear(hidden_dim, output_dim))
        # init net weights and biases
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1)
                torch.nn.init.constant_(m.bias, val=0)
        # get numel learnable
        self.numel_learnable = 0
        named_param_list = list(self.named_parameters())
        for name, param in named_param_list:
            self.numel_learnable += torch.numel(param)

    def forward(self, x):
        # fixme several steps for debugging
        out0 = self.net[0](x)  # linear first layer
        out1 = self.net[1](out0)  # input-batch-norm layer
        out2 = self.net[2](out1)  # tanh layer
        out3 = self.net[3](out2)  # 2nd linear layer
        # fixme : end debugging code
        out = self.net(x)
        return out

    def __str__(self):
        return f"\n***\n" \
               f"NN-Model \n{str(self.net)}\n" \
               f"numel_learnable = {self.numel_learnable}\n" \
               f"***"


class PolyReg(torch.nn.Module):
    def __init__(self, in_dim, out_dim, deg):
        super().__init__()
        self.deg = deg
        self.linear_part = torch.nn.Linear(in_features=in_dim * (deg + 1), out_features=out_dim)

    def forward(self, x):
        x_max = torch.max(x)
        x_pows = []
        for d in range(self.deg + 1):
            x_pows.append(torch.pow(x, d))

        x_pows_cat = torch.cat(x_pows, dim=1)
        x_max = torch.max(x_pows_cat)
        y_hat = self.linear_part(x_pows_cat)
        return y_hat


class LinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = 10
        self.lin_model = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim))
        # torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.lin_model(x)


#########################
def get_param_grad_norm_sum(param_list: List[torch.nn.Parameter]):
    s = sum(map(lambda x: torch.norm(x.grad), param_list))
    return s


def get_param_grad_norm_avg(param_list: List[torch.nn.Parameter]):
    sum_ = get_param_grad_norm_sum(param_list)
    num_el = sum(map(lambda x: torch.numel(x), param_list))
    avg_ = float(sum_) / num_el
    return avg_


def vanilla_opt_block(model, X, y, optim, loss_func, output_norm: str, y_train_mean,
                      y_train_std):
    optim.zero_grad()
    # TODO revisit output-normalization
    #   ref : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
    #   quote :
    """
    Scaling input and output variables is a critical step in using neural 
    network models.
    In practice it is nearly always advantageous to apply pre-processing 
    transformations to the input data before it is presented to a network. 
    >>> Similarly, the outputs of the network are often post-processed to 
    give the required output values.
    """
    #  TODO (Notes)
    #   Should we scale the output target variable
    #   1. https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    # quote :
    """
    # generate regression dataset
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
    Neural networks generally perform better when the real-valued input and output variables are to be scaled to a sensible range. For this problem, each of the input variables and the target variable have a Gaussian distribution; therefore, standardizing the data in this case is desirable.
    
    We can achieve this using the StandardScaler transformer class also from the scikit-learn library. On a real problem, we would prepare the scaler on the training dataset and apply it to the train and test sets, but for simplicity, we will scale all of the data together before splitting into train and test sets.
    
    # standardize dataset
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
    # standardize dataset
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
    Once scaled, the data will be split evenly into train and test sets.
    """
    # TODO (Notes-Cont.)
    #   Should we scale output-target variable
    #   - https://datascience.stackexchange.com/questions/24214/why-should-i-normalize-also-the-output-data/24218#24218
    #   - http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
    #   Search for "Search for Should I standardize the target variables (column vectors)? in above page."
    #   - Another opinion
    #     https://stackoverflow.com/questions/42604261/neural-networks-normalizing-output-data
    if output_norm == 'batch':
        Dy = y.size()[1]
        target_batch_norm = \
            torch.nn.BatchNorm1d(num_features=Dy, affine=False,
                                 track_running_stats=False)
        # not affine, not learnable . NEVER set affine=true for output
        # standardization
        y_norm = target_batch_norm(y)
    elif output_norm == 'data':
        y_norm = (y - y_train_mean) / y_train_std
    elif output_norm is None:
        y_norm = y
    else:
        raise ValueError(f'invalid out_norm flag {output_norm}')

    y_hat = model(X)
    # make_dot(y_hat, params=dict(model.named_parameters())).render(str(type(model)), format="png")
    loss = loss_func(y_hat, y_norm)
    loss.backward()
    # fixme , debug code
    # param_list = list(model.parameters())
    # grad_norm_sum = get_param_grad_norm_sum(param_list)
    # grad_norm_avg = get_param_grad_norm_avg(param_list)

    # TODO : gradient clipping, revisit
    #   - https://stackoverflow.com/questions/67236480/why-is-the-clip-grad-norm-function-used-here
    #   - (paper) http://proceedings.mlr.press/v28/pascanu13.pdf
    #   - https://gist.github.com/omairaasim/2fb6419226be21c354e17c6d587bf1a4
    torch.nn.utils.clip_grad_norm(parameters=model.parameters(),
                                  max_norm=grad_clip_max_norm)
    optim.step()
    return loss.item()


# fixme, refactor to be as vanilla_opt_block

def tt_opt_block(model, X_train, Y_train, optim, loss_func):
    raise NotImplementedError("Must refactor to be as vanilla_opt_block")
    # optim.zero_grad()
    # Y_hat = model.forward_old(X_train)
    # make_dot(Y_hat, params=dict(model.named_parameters())).render(str(type(model)), format="png")
    # loss = loss_func(Y_hat, Y_train)
    # loss.backward()
    # param_list = list(model.parameters())
    # grad_norm_sum = get_param_grad_norm_sum(param_list)
    # grad_norm_avg = get_param_grad_norm_avg(param_list)
    # optim.step()
    # return loss.item()


LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)10s()] %(asctime)s %(levelname)s %(message)s"
DATE_TIME_FORMAT = "%Y-%m-%d:%H:%M:%S"
SEEDS = [42, 18819191, 71623183, 71623183, 12345, 54321, 987654321]
if __name__ == '__main__':
    ### set logging configs
    pd.set_option('display.max_rows', None)
    time_stamp = datetime.now().strftime(DATE_TIME_FORMAT)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    fh = logging.FileHandler(filename=f"./experiments_log/experiment_{time_stamp}.log")
    formatter = logging.Formatter(fmt=LOG_FORMAT)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    # set float precision decimals
    precision_decimals = 8
    ### set seed
    # Why 42 ? https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a
    SEED = SEEDS[0]
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    logger.info(f'SEED = {SEED}')
    ### Training ####
    # general params
    batch_size = 32
    input_dim = 3
    output_dim = 3
    loss_fn = torch.nn.MSELoss()
    train_epochs = 10000
    epochs_losses_window = 10
    input_batch_norm = False
    output_norm = None  # can be "data","batch" or None
    grad_clip_max_norm = 10
    # opts
    lr = 2e-1
    sgd_momentum = 0.99
    # schedulers
    linear_lr_scheduler_start_factor = 1.0
    linear_lr_scheduler_end_factor = 1e-4
    linear_lr_scheduler_total_iter = int(0.9 * train_epochs)
    #
    reduce_on_plateau_sch_factor = 0.8
    reduce_on_plateau_sch_min_lr = 1e-3
    # tt
    tt_poly_deg = 10
    tt_rank = 3
    tt_order = 4
    # rbf
    rbf_n_centres = 16
    kernel_name = "gaussian"
    # nn
    nn_hidden_dim = 50
    ### Data
    N_train = 1000
    N_test = N_train
    x_gen_norm_mean = 0
    x_gen_norm_std = 1
    # vdp
    vdp_mio = 0.5
    # lorenz
    rho = 28
    sigma = 10
    beta = 8 / 3
    #
    normalize_data_source_X_train = False
    normalize_data_source_Y_train = False
    normalize_data_source_X_test = False
    normalize_data_source_Y_test = False

    ## Models ##
    # => Set model here
    # - Main models for now
    # model = NNmodel(input_dim=input_dim, hidden_dim=nn_hidden_dim,
    #                 output_dim=output_dim,
    #                 input_batch_norm=input_batch_norm)
    # model = TTpoly2in2out(rank=rank, deg=poly_deg)
    # model = RBFN(in_dim=input_dim, out_dim=output_dim, n_centres=rbf_n_centres, basis_fn_str=kernel_name,
    #              input_batch_norm=input_batch_norm)
    model = TTRBF(in_dim=input_dim, out_dim=output_dim,
                  num_rbf_centers=rbf_n_centres, rbf_basis_function_name=kernel_name,
                  tt_rank=tt_rank, order=tt_order)
    # ---
    # - some sandbox models
    # model = LinearModel(in_dim=Dx, out_dim=1)
    # model = TensorTrainFixedRank(dims=[poly_deg + 1] * Dx, fixed_rank=rank, requires_grad=True, unif_low=-0.01,
    #                              unif_high=0.01, poly_deg=poly_deg)
    # model = PolyReg(in_dim=Dx, out_dim=output_dim, deg=5)
    # model = LinearModeEinSum(in_dim=Dx, out_dim=1)
    # model = PolyLinearEinsum(in_dim=Dx, out_dim=output_dim, deg=poly_deg)
    # model = TTpoly4dim(in_dim=Dx, out_dim=output_dim, deg=3, rank=2)
    # model = FullTensorPoly4dim(input_dim=Dx, out_dim=output_dim, deg=poly_deg)
    # model = TTpoly1dim(in_dim=1, out_dim=1, deg=5)
    # model = TTpoly2dim(in_dim=Dx, out_dim=1, deg=poly_deg, rank=3)

    ### Optimizers ####
    # TODO (SGD with momentum)
    #   read https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    #   and  http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    # , nesterov=True, momentum=0.99
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(params=model.parameters(),lr=lr,momentum=0.99)
    logger.info(f'model = {model}')
    logger.info(f'optimizer  = {optimizer}')
    # lr scheduler
    # lr_scheduler = LinearLR(optimizer=optimizer,
    #                         start_factor=linear_lr_scheduler_start_factor,
    #                         end_factor=linear_lr_scheduler_end_factor,
    #                         total_iters=linear_lr_scheduler_total_iter)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=reduce_on_plateau_sch_factor,
                                     min_lr=reduce_on_plateau_sch_min_lr)
    logger.info(f'lr_scheduler = {lr_scheduler}')

    ### data #####
    logger.info(f'Normalize-Data-source-X-train = {normalize_data_source_X_train}')
    logger.info(f'Normalize-Data-source-Y-train = {normalize_data_source_Y_train}')
    logger.info(f'Normalize-Data-source-X-test = {normalize_data_source_X_test}')
    logger.info(f'Normalize-Data-source-Y-test = {normalize_data_source_Y_test}')
    # data_set = ToyData1(input_dim=input_dim,N=N_samples_data)

    # train_data_set = VDP(mio=vdp_mio, N=N_train,
    #                      x_gen_norm_mean=x_gen_norm_mean,
    #                      x_gen_norm_std=x_gen_norm_std,
    #                      normalize_X=normalize_data_source_X_train,
    #                      normalize_Y=normalize_data_source_Y_train,
    #                      train_or_test="train")
    train_data_set = LorenzSystem(N=N_train, rho=rho, sigma=sigma, beta=beta, x_gen_norm_mean=x_gen_norm_mean,
                                  x_gen_norm_std=x_gen_norm_std, normalize_X=normalize_data_source_X_train,
                                  normalize_Y=normalize_data_source_Y_train, train_or_test="train")
    if isinstance(train_data_set, VDP):
        assert input_dim == 2
        assert output_dim == 2
    elif isinstance(train_data_set, LorenzSystem):
        assert input_dim == 3
        assert output_dim == 3
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
    # test_data_set = VDP(mio=vdp_mio, N=N_test, x_gen_norm_mean=x_gen_norm_mean,
    #                     x_gen_norm_std=x_gen_norm_std,
    #                     normalize_X=normalize_data_source_X_test,
    #                     normalize_Y=normalize_data_source_Y_test,
    #                     train_or_test="test")
    test_data_set = LorenzSystem(N=N_test, rho=rho, sigma=sigma, beta=beta, x_gen_norm_mean=x_gen_norm_mean,
                                 x_gen_norm_std=x_gen_norm_std, normalize_X=normalize_data_source_X_train,
                                 normalize_Y=normalize_data_source_Y_train, train_or_test="test")
    test_data_loader = DataLoader(dataset=test_data_set, batch_size=batch_size, shuffle=True)

    logger.info(f'train-dataset = {train_data_set}')
    logger.info(f'test-dataset = {test_data_set}')
    logger.info(f'train-epochs = {train_epochs}')

    start_time_stamp = datetime.now()
    # training
    logger.info(f'Input batch normalization = {input_batch_norm}')
    logger.info(f'Output Normalization = {output_norm}')
    logger.info(f'Gradient-clipping max-norm = {grad_clip_max_norm}')
    #
    logger.info(f'epochs_losses_window = {epochs_losses_window}')
    epochs_losses = []
    epochs_losses_curve_x = []
    epochs_losses_curve_y = []
    for train_epoch in tqdm(range(train_epochs + 1), desc='train-epochs'):
        batches_losses = []
        for i, (X_train, Y_train) in enumerate(train_data_loader):

            if isinstance(model, (
                    NNmodel, LinearModel, PolyReg, LinearModeEinSum, TTpoly4dim, FullTensorPoly4dim, TTpoly1dim,
                    TTpoly2dim, TTpoly2in2out, RBFN, TTRBF)):
                loss_val = vanilla_opt_block(model=model, X=X_train, y=Y_train, optim=optimizer,
                                             loss_func=loss_fn, y_train_mean=train_data_set.get_Y_mean(),
                                             y_train_std=train_data_set.get_Y_std(), output_norm=output_norm)
            elif isinstance(model, TensorTrainFixedRank):
                loss_val = tt_opt_block(model=model, X_train=X_train, Y_train=Y_train, optim=optimizer,
                                        loss_func=loss_fn)
                assert (not np.isnan(loss_val)) and (not np.isinf(loss_val))
            else:
                raise ValueError(f"Error {type(model)}")
            # todo : read https://stackoverflow.com/questions/54053868/how-do-i-get-a-loss-per-epoch-and-not-per-batch
            batches_losses.append(loss_val)
            # print('epoch {}, batch {}, loss {}'.format(epoch, i, np.nanmean(losses)))
        ### code for epoch ###
        # scheduler lr update
        before_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step(np.nanmean(batches_losses))
        after_lr = optimizer.param_groups[0]["lr"]
        logger.info(f'epoch # {train_epoch} - '
                    f'for optimizer {type(optimizer)} -'
                    f'lr : {np.round(before_lr, precision_decimals)}->'
                    f' {np.round(after_lr, precision_decimals)} -'
                    f'loss = {np.mean(batches_losses)}')
        # log epoch loss
        epochs_losses.append(np.mean(batches_losses))
        if train_epoch >= epochs_losses_window and train_epoch % epochs_losses_window == 0:
            assert len(epochs_losses) >= epochs_losses_window
            epochs_rolling_window_losses = epochs_losses[-epochs_losses_window:]
            # fixme : I am using mean not nan-mean to capture any nan loss, but should
            #   check for it explicitly
            logger.info(f'*** epoch {train_epoch}, rolling-avg-loss (window={epochs_losses_window})= '
                        f'{np.mean(epochs_rolling_window_losses)}')
            epochs_losses_curve_x.append(train_epoch)
            epochs_losses_curve_y.append(np.mean(epochs_rolling_window_losses))

    # train time logging
    end_time_stamp = datetime.now()
    training_time_sec = (end_time_stamp - start_time_stamp).seconds
    logger.info(f'training time in seconds = {training_time_sec}')
    # epoch loss curve
    train_epoch_loss_df = pd.DataFrame({'epochs': epochs_losses_curve_x,
                                        'rolling-avg-loss': epochs_losses_curve_y})
    logger.info('train-epochs-loss curve df :')
    logger.info(f"\n{train_epoch_loss_df}")
    ### Log model params after training ###
    logger.info('Model parameters after training')
    logger.info(f'Model = {type(model).__name__}')
    for name, param in model.named_parameters():
        logger.info(f'{name} = {param}')

    ### Out of Sample Testing ###
    logger.info(f'Out-of sample batch-test')
    for j, (X_test, Y_test) in enumerate(test_data_loader):
        Y_hat_test = model(X_test)
        Y_hat_denorm = Y_hat_test  # * train_data_set.get_Y_std() + train_data_set.get_Y_mean()
        loss_test = loss_fn(Y_hat_denorm, Y_test)
        logger.info(f'test-batch  # {j} => test-loss = {loss_test.item()}')
"""
scratch-pad
vanilla ref steps
# optimizer.zero_grad()
# # Forward to get output
# y_hat = model(X)
# # Calculate Loss
# loss = loss_fn(y_hat, y)
#
# # Getting gradients w.r.t. parameters
# loss.backward()
# # Updating parameters
# optimizer.step()
"""
