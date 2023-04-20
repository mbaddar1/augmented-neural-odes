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
"""

import logging

# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
from typing import List
from datetime import datetime
import torch

torch.use_deterministic_algorithms(True)
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot
import random
from phd_experiments.hybrid_node_tt.models import TensorTrainFixedRank


class VDP(Dataset):
    # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
    # https://www.johndcook.com/blog/2019/12/22/van-der-pol/
    # https://arxiv.org/pdf/0803.1658.pdf
    # todo add plotting
    def __init__(self, mio: float, N: int):
        self.N = N
        Dx_vdp = 2
        self.mio = mio
        self.X = torch.distributions.Normal(loc=0, scale=5).sample(torch.Size([self.N, Dx_vdp]))
        x1 = self.X[:, 0].view(-1, 1)
        x2 = self.X[:, 1].view(-1, 1)
        x1_dot = x2
        x2_dot = mio * (1 - x1 ** 2) * x2 - x1
        self.Y = torch.cat([x1_dot, x2_dot], dim=1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __str__(self):
        return f"***\nVDP Dataset\nmio = {self.mio}\n***"


class LorenzSystem(Dataset):
    # High Dim non-linear systems
    # https://tglab.princeton.edu/wp-content/uploads/2011/03/Mol410Lecture13.pdf (P 2)
    # https://en.wikipedia.org/wiki/Lorenz_system
    pass


class FVDP(Dataset):
    # http://math.colgate.edu/~wweckesser/pubs/FVDPI.pdf
    pass


class ToyData1(Dataset):
    def __init__(self, Dx):
        N = 10000
        self.N = N
        W_dict = {1: torch.tensor([0.1]).view(1, 1),
                  2: torch.tensor([0.1, -0.2]).view(1, 2),
                  4: torch.tensor([0.1, -0.2, 0.3, -0.8]).view(1, 4)}
        W = W_dict[Dx]
        self.X = torch.distributions.Normal(0, 1).sample(torch.Size([N, Dx]))
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
class TTrbf(torch.nn.Module):
    # https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
    pass


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
        self.numel_learnable = self.G0.numel() + self.G1.numel()
        self.deg = deg

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
    def __init__(self, input_dim, hidden_dim, output_dim):

        super(NNmodel, self).__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden_dim, output_dim))
        self.numel_learnable = 0
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1.0)
                self.numel_learnable += m.weight.numel()
                torch.nn.init.constant_(m.bias, val=0.0)
                self.numel_learnable += m.bias.numel()

    def forward(self, x):
        out = self.net(x)
        return out

    def __str__(self):
        return f"\n*** NNmodel \n {str(self.net)}\nnumel_learnable = {self.numel_learnable}\n***"


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


LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)10s()] %(asctime)s %(levelname)s %(message)s"
DATE_TIME_FORMAT = "%Y-%m-%d:%H:%M:%S"
SEEDS = [42, 18819191, 71623183, 71623183, 12345, 54321, 987654321]
if __name__ == '__main__':

    ### set logging
    time_stamp = datetime.now().strftime(DATE_TIME_FORMAT)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    fh = logging.FileHandler(filename=f"./experiments_log/experiment_{time_stamp}.log")
    formatter = logging.Formatter(fmt=LOG_FORMAT)
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    ## set seed
    # Why 42 ? https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a
    SEED = SEEDS[0]
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    logger.info(f'SEED = {SEED}')
    ### train params ####
    batch_size = 128
    Dx = 4
    output_dim = 1
    poly_deg = 10
    rank = 3
    loss_fn = torch.nn.MSELoss()
    lr = 0.01
    epochs = 1000
    vdp_mio = 0.5
    N_samples_data = 100
    nn_hidden_dim = 500
    ## Models ##
    # - Main models for now
    # model = NNmodel(input_dim=2, hidden_dim=nn_hidden_dim, output_dim=2)
    model = TTpoly2in2out(rank=rank, deg=poly_deg)
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
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, nesterov=True, momentum=0.1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0)
    logger.info(f'model = {model}')
    logger.info(f'opt  = {optimizer}')

    # opt
    ### data #####
    data_set = VDP(mio=vdp_mio, N=N_samples_data)
    dl = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
    logger.info(f'data = {data_set}')
    logger.info(f'epochs = {epochs}')

    start_time_stamp = datetime.now()
    # training
    for epoch in range(epochs + 1):
        # Clear gradients w.r.t. parameters
        losses = []
        for i, (X, Y) in enumerate(dl):
            # X_max = torch.max(X)
            # X_min = torch.min(X)
            # TODO, is sigmoid a good way to normalize data ?
            X_norm = torch.nn.Sigmoid()(X)  # (X - X_min) / (X_max - X_min)
            if isinstance(model, (
                    NNmodel, LinearModel, PolyReg, LinearModeEinSum, TTpoly4dim, FullTensorPoly4dim, TTpoly1dim,
                    TTpoly2dim, TTpoly2in2out)):
                loss_val = nn_opt_block(model=model, X=X_norm, y=Y, optim=optimizer, loss_func=loss_fn)
            elif isinstance(model, TensorTrainFixedRank):
                loss_val = tt_opt_block(model=model, X=X_norm, y=Y, optim=optimizer, loss_func=loss_fn)
            else:
                raise ValueError(f"Error {type(model)}")

            losses.append(loss_val)
            # print('epoch {}, batch {}, loss {}'.format(epoch, i, np.nanmean(losses)))
        if epoch % 10 == 0:
            logger.info(f'epoch {epoch}, loss = {np.nanmean(losses)}')
    end_time_stamp = datetime.now()
    training_time_sec = (end_time_stamp - start_time_stamp).seconds
    logger.info(f'training time in seconds = {training_time_sec}')

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
