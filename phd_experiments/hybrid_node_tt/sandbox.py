# https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from phd_experiments.hybrid_node_tt.models import TensorTrainFixedRank


class ToyData1(Dataset):
    def __init__(self, Dx):
        N = 10000
        self.N = N
        W = torch.tensor([0.1, -0.2, 0.3]).view(1, Dx)
        X = torch.distributions.Normal(0, 1).sample(torch.Size([N, Dx]))
        X_nl = torch.sin(X)
        y = torch.einsum('ij,bj->b', W, X_nl).view(-1, 1)
        self.X = X_nl
        self.Y = y

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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
        self.lin_model = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden_dim),
                                             torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.lin_model(x)


def nn_opt_block(model, X, y, optim, loss_func):
    optim.zero_grad()
    y_hat = model(X)
    loss = loss_func(y_hat, y)
    loss.backward()
    optim.step()
    return loss.item()


def tt_opt_block(model, X, y, optim, loss_func):
    optim.zero_grad()
    y_hat = model.forward_old(X)
    loss = loss_func(y_hat, y)
    loss.backward()
    optim.step()
    return loss.item()


if __name__ == '__main__':
    Dx = 3
    output_dim = 1
    poly_deg = 3
    rank = 3
    #
    # model = NNmodel(Dx, output_dim)
    # model = LinearModel(in_dim=Dx, out_dim=1)
    # model = TensorTrainFixedRank(dims=[poly_deg + 1] * Dx, fixed_rank=rank, requires_grad=True, unif_low=-0.01,
    #                              unif_high=0.01, poly_deg=poly_deg)
    model = PolyReg(in_dim=Dx,out_dim=output_dim,deg=poly_deg)
    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ds = ToyData1(Dx)
    dl = DataLoader(dataset=ds, batch_size=64, shuffle=True)
    epochs = 50000
    print(f'model type = {type(model)}')
    for epoch in range(epochs):
        # Clear gradients w.r.t. parameters
        losses = []
        for i, (X, y) in enumerate(dl):
            if isinstance(model, (NNmodel, LinearModel,PolyReg)):
                loss_val = nn_opt_block(model=model, X=X, y=y, optim=optimizer, loss_func=loss_fn)
            elif isinstance(model, TensorTrainFixedRank):
                loss_val = tt_opt_block(model=model, X=X, y=y, optim=optimizer, loss_func=loss_fn)
            else:
                raise ValueError(f"Errooooor {type(model)}")

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
