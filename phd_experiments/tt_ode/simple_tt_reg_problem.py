"""
The purpose of this script is to experiment
1- ALS training
2- Gradient descent training ( as a base-line)
for a simple sklearn regression problem
Why? I want to set a several baselines
1- TT-ODE trained by normal Backprop-Gradient-Descent
2- TT-ODE trained by Adjoint-Sensitivity

refs
https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.nn
from torch import Tensor
from torch.nn import MSELoss
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.datasets import load_diabetes
from phd_experiments.tt_ode.ttode_model import TensorTrainODEBLOCK


class SyntheticRegressionDataSet1(Dataset):
    def __init__(self, N: int, A: torch.Tensor, b: torch.Tensor, u_low: float, u_high):
        assert len(list(A.size())) == 2, "A must be of dim 2"
        assert len(list(b.size())) == 2 and list(b.size())[
            0] == 1, "b must be of 2 dims of 1xDy"  # 1 vector dim of Dyx1

        assert list(A.size())[0] == list(b.size())[1], "A second dim must == b first one"

        self.N = N
        # syn_model = torch.nn.Linear(in_features=list(A.size())[0], out_features=list(A.size())[1])
        # with torch.no_grad():
        #     syn_model.weight = torch.nn.Parameter(A)
        #     syn_model.bias = torch.nn.Parameter(b)
        X = torch.distributions.Uniform(low=u_low, high=u_high).sample(sample_shape=torch.Size([N, list(A.size())[1]]))
        tmp = torch.einsum('bi,ji->bj', X, A)
        Y = tmp + b.repeat(N, 1)
        # Y = syn_model(X)
        self.x_train = X
        self.y_train = Y

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class SklearnDiabetesDataset(Dataset):

    def __init__(self, dtype: torch.dtype):
        X, y = load_diabetes(return_X_y=True, as_frame=True)

        self.x_train = torch.tensor(torch.Tensor(X.values), dtype=dtype)
        self.y_train = torch.tensor(torch.Tensor(y.values), dtype=dtype)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class NN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim):
        super().__init__()
        # self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Linear(hidden_dim, hidden_dim),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Linear(hidden_dim, hidden_dim),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Linear(hidden_dim, out_dim))
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, out_dim))

    def forward(self, x: Tensor):
        y = self.model(x)
        return y


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger()
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    #
    # logger.info(f'X shape = {X.shape}')
    # logger.info(f'Y shape = {y.shape}')
    # training params

    epochs = 100
    batch_size = 64
    hidden_dim = 256
    model_type = 'tt'
    dtype = torch.float64
    # training
    # ds = SklearnDiabetesDataset(dtype=dtype)
    A = torch.Tensor([1.0, -2.0, 3.0]).view(1, 3)
    b = torch.Tensor([10.0]).T.view(1, 1)
    ds = SyntheticRegressionDataSet1(N=1024, A=A, u_low=-10.0, u_high=10.0, b=b)
    train_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    model = None
    if model_type == 'nn':
        model = NN(input_dim=3, out_dim=1, hidden_dim=hidden_dim)
    elif model_type == 'tt':
        model = TensorTrainODEBLOCK(input_dimensions=[3], output_dimensions=[1], tensor_dimensions=[4],
                                    basis_str='poly,3', t_span=(0, 0.4), non_linearity=None, t_eval=None,
                                    forward_impl_method='ttode_als', tt_rank=3, custom_autograd_fn=True)
    else:
        raise ValueError(f"unknown model type {model_type}")
    parameters_ = model.parameters()
    parameters_list = list(parameters_)
    # if model_type == 'tt':
    #     assert len(parameters_list) == X.shape[1], "Number of cores (tt parameters) must = data_dim"
    optimizer = Adam(params=parameters_list, lr=0.1)
    mse_fn = MSELoss()
    epochs_avg_losses = []
    for epoch in tqdm(range(epochs), desc='epoch'):
        epoch_losses = []
        for batch_idx, (X, y) in enumerate(train_loader):
            for batch_repeat_idx in range(0, 10):  # run over the same batch many times
                optimizer.zero_grad()
                # y_hat = None
                y_hat = model(X)
                # if model_type in ['nn']:
                #     y_hat = model(X)
                # elif model_type == 'tt':
                #     Phi = Basis.poly(x=X, t=None, poly_deg=basis_poly_deg)
                #     y_hat = model(Phi)
                assert y_hat is not None, "y_hat cannot be None"
                mse_val = mse_fn(y, y_hat)
                rmse_val = torch.sqrt(mse_val)
                epoch_losses.append(rmse_val.item())
                mse_val.backward()
                optimizer.step()
                logger.info(
                    f'epoch {epoch}|\t batch {batch_idx}|\t  batch-repeat {batch_repeat_idx}|\tloss = {rmse_val.item()}')
                logger.info('==================')
                # time.sleep(1)
        epochs_avg_losses.append(np.nanmean(epoch_losses))
        if epoch % 10 == 0:
            pass
            # logger.info(f'Epoch : {epoch} => rmse = {epoch_losses[-1]}')

    assert len(epochs_avg_losses) == epochs
    print(len(epochs_avg_losses))
    # TODO compare results with
    #   https://www.kaggle.com/code/rahulrajpandey31/diabetes-analysis-linear-reg-from-scratch/notebook
    logger.info('training finished')
    fig = plt.plot(epochs_avg_losses)
    plt.savefig(f'convergence_{model_type}.png')
