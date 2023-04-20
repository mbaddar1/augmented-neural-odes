"""
Bad Gradient in pytorch
https://gist.github.com/apaszke/f93a377244be9bfcb96d3547b9bc424d
https://discuss.pytorch.org/t/how-to-check-for-vanishing-exploding-gradients/9019/3

torch einsum autograd
https://discuss.pytorch.org/t/automatic-differentation-for-pytorch-einsum/112504
"""

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
from typing import List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot

from phd_experiments.hybrid_node_tt.models import TensorTrainFixedRank


class ToyData1(Dataset):
    def __init__(self, Dx):
        # fixme one special case
        assert Dx == 4
        N = 10000
        self.N = N
        W = torch.tensor([0.1, -0.2, 0.3, -0.8]).view(1, Dx)
        X = torch.distributions.Normal(0, 1).sample(torch.Size([N, Dx]))
        X_nl = torch.sin(X)
        y = torch.einsum('ij,bj->b', W, X_nl).view(-1, 1)
        self.X = X_nl
        self.Y = y

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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
        u_low = -0.05
        u_high = 0.05
        self.G0 = torch.nn.Parameter(torch.distributions.Uniform(u_low, u_high).sample(torch.Size([deg + 1, rank])))
        self.G1 = torch.nn.Parameter(
            torch.distributions.Uniform(u_low, u_high).sample(torch.Size([rank, deg + 1, rank])))
        self.G2 = torch.nn.Parameter(
            torch.distributions.Uniform(u_low, u_high).sample(torch.Size([rank, deg + 1, rank])))
        self.G3 = torch.nn.Parameter(torch.distributions.Uniform(u_low, u_high).sample(torch.Size([rank, deg + 1])))

    def forward(self, X):
        # generate Phi
        Phi = []
        b = X.size()[0]
        dim = X.size()[1]
        for dim_idx in range(dim):
            X_d = X[:, dim_idx]
            x_list = []
            for deg_idx in range(self.deg + 1):
                x_list.append(torch.pow(X_d, deg_idx).view(-1, 1))
            Phi_tensor = torch.cat(x_list, dim=1)
            Phi.append(Phi_tensor)

        einsum_params = [self.G0, Phi[0], self.G1, Phi[1], self.G2, Phi[2], self.G3, Phi[3]]
        # ValueError: Size of label 'i' for operand 6 (3) does not match previous terms (4).
        einsum_str = "ac,ba,cfe,bf,ehi,bh,iq,bq->b"
        res = torch.einsum(einsum_str,einsum_params)
        return res.view(-1,1)

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


class NNmodel(nn.Module):
    def __init__(self, input_dim, output_dim):
        hidden_dim = 50
        super(NNmodel, self).__init__()
        self.net = torch.nn.Sequential(nn.Linear(input_dim, hidden_dim), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden_dim, output_dim))
        # self.net = torch.nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        out = self.net(x)
        return out


class PolyReg(nn.Module):
    def __init__(self, in_dim, out_dim, deg):
        super().__init__()
        self.deg = deg
        self.linear_part = torch.nn.Linear(in_features=in_dim * (deg + 1), out_features=out_dim)

    def forward(self, x):
        x_pows = []
        for d in range(self.deg + 1):
            x_pows.append(torch.pow(x, d))
        x_pows_cat = torch.cat(x_pows, dim=1)
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


def get_param_grad_norm_sum(param_list: List[torch.nn.Parameter]):
    s = sum(map(lambda x: torch.norm(x.grad), param_list))
    return s


def get_param_grad_norm_avg(param_list: List[torch.nn.Parameter]):
    sum_ = get_param_grad_norm_sum(param_list)
    num_el = sum(map(lambda x: torch.numel(x), param_list))
    avg_ = float(sum_) / num_el
    return avg_


def nn_opt_block(model, X, y, optim, loss_func):
    optim.zero_grad()
    y_hat = model(X)
    make_dot(y_hat, params=dict(model.named_parameters())).render(str(type(model)), format="png")
    loss = loss_func(y_hat, y)
    loss.backward()
    param_list = list(model.parameters())
    grad_norm_sum = get_param_grad_norm_sum(param_list)
    grad_norm_avg = get_param_grad_norm_avg(param_list)
    optim.step()
    return loss.item()


def tt_opt_block(model, X, y, optim, loss_func):
    optim.zero_grad()
    y_hat = model.forward_old(X)
    make_dot(y_hat, params=dict(model.named_parameters())).render(str(type(model)), format="png")
    loss = loss_func(y_hat, y)
    loss.backward()
    param_list = list(model.parameters())
    grad_norm_sum = get_param_grad_norm_sum(param_list)
    grad_norm_avg = get_param_grad_norm_avg(param_list)
    optim.step()
    return loss.item()


if __name__ == '__main__':
    Dx = 4
    output_dim = 1
    poly_deg = 3
    rank = 3
    loss_fn = nn.MSELoss()
    lr = 0.01
    epochs = 50000
    rank = 3
    ## Models ##
    #
    # model = NNmodel(Dx, output_dim)
    # model = LinearModel(in_dim=Dx, out_dim=1)
    # model = TensorTrainFixedRank(dims=[poly_deg + 1] * Dx, fixed_rank=rank, requires_grad=True, unif_low=-0.01,
    #                              unif_high=0.01, poly_deg=poly_deg)
    # model = PolyReg(in_dim=Dx, out_dim=output_dim, deg=poly_deg)
    # model = LinearModeEinSum(in_dim=Dx, out_dim=1)
    # model = PolyLinearEinsum(in_dim=Dx, out_dim=output_dim, deg=poly_deg)
    model = TTpoly4dim(in_dim=Dx, out_dim=output_dim, deg=poly_deg, rank=3)
    ###########################
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    ds = ToyData1(Dx)
    dl = DataLoader(dataset=ds, batch_size=64, shuffle=True)

    print(f'model type = {type(model)}')
    for epoch in range(epochs):
        # Clear gradients w.r.t. parameters
        losses = []
        for i, (X, y) in enumerate(dl):
            if isinstance(model, (NNmodel, LinearModel, PolyReg, LinearModeEinSum, TTpoly4dim)):
                loss_val = nn_opt_block(model=model, X=X, y=y, optim=optimizer, loss_func=loss_fn)
            elif isinstance(model, TensorTrainFixedRank):
                loss_val = tt_opt_block(model=model, X=X, y=y, optim=optimizer, loss_func=loss_fn)
            else:
                raise ValueError(f"Error {type(model)}")

            losses.append(loss_val)
            print('epoch {}, batch {}, loss {}'.format(epoch, i, np.nanmean(losses)))

"""
scratch-pad
vanilla steps
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
