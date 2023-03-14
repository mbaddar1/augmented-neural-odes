"""
From scratch impl.
https://d2l.ai/chapter_convolutional-modern/resnet.html
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
"""
import logging

import numpy as np
import sklearn.model_selection
import torch.nn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from phd_experiments.datasets.torch_sklearn_diabetes import TorchDiabetesDataset
from phd_experiments.datasets.toy_linear import ToyLinearDataSet1
from phd_experiments.datasets.toy_relu import ToyRelu


def build_sklearn_diabetes_baseline(X: np.ndarray, y: np.ndarray, loss_function_name: str):
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    epochs = 100
    loss_vals_list = []
    for i in range(epochs):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        if loss_function_name == 'mse':
            loss_val = mean_squared_error(y_test, y_pred)
            loss_vals_list.append(loss_val)

    ret = {'model': str(regr.__class__), 'loss_function_name': loss_function_name,
           'avg-loss': np.nanmean(loss_vals_list), 'std-loss': np.nanstd(loss_vals_list)}
    return ret


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.model = torch.nn.Linear(in_features=input_dim, out_features=out_dim)

    def forward(self, x):
        return self.model(x)


class NNBasic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, n_layers, activation_function_name):
        super().__init__()
        assert n_layers >= 3, "n_layers must >=3"
        self.model = Sequential()
        self.model.append(torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)) \
            .append(get_activation_model(activation_function_name))
        for _ in range(n_layers - 2):
            self.model.append(torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)) \
                .append(get_activation_model(activation_function_name))
        self.model.append(torch.nn.Linear(in_features=hidden_dim, out_features=out_dim))  # .append(torch.nn.Identity())

    def forward(self, x):
        return self.model(x)


def get_activation_model(activation_function_name: str):
    if activation_function_name == 'relu':
        return torch.nn.ReLU()
    if activation_function_name == 'tanh':
        return torch.nn.Tanh()
    elif activation_function_name == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation_function_name == 'identity':
        return torch.nn.Identity()
    else:
        raise ValueError(f'Unknown activation name = {activation_function_name}')


class ResNetNonLinearSubBlock(torch.nn.Module):
    """
    for residual block f(x) = g(x)+x , this class implements g(x)
    """

    def __init__(self, dim, activation_function_name):
        super().__init__()
        # https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
        # https://d2l.ai/_images/residual-block.svg

        self.model = Sequential(torch.nn.Linear(in_features=dim, out_features=dim),
                                get_activation_model(activation_function_name),
                                torch.nn.Linear(in_features=dim, out_features=dim))

    def forward(self, x: torch.Tensor):
        return self.model(x)


class ResNetBlock(torch.nn.Module):
    def __init__(self, dim, activation_function_name):
        # https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
        # https://d2l.ai/_images/residual-block.svg
        super().__init__()
        self.non_linear_block = ResNetNonLinearSubBlock(dim=dim, activation_function_name=activation_function_name)

    def forward(self, x):
        return x + self.non_linear_block(x)  # f(x) = g(x) + x


class ResNetBasic(torch.nn.Module):
    """
    https://d2l.ai/chapter_convolutional-modern/resnet.html#resnet-model
    https://d2l.ai/_images/resnet18-90.svg
    """

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int,
                 activation_function_name: str):
        super().__init__()
        # a bit of modification on the arch in https://d2l.ai/chapter_convolutional-modern/resnet.html#resnet-model
        # activation_model = get_activation_model(activation_function_name=activation_function_name)
        self.model = Sequential()
        # add init model
        self.model.append(torch.nn.Linear(input_dim, hidden_dim)).append(get_activation_model(activation_function_name))
        # add baseline blocks
        assert n_layers >= 3, "Number of layers must be >=3"
        n_resnet_blocks = n_layers - 2
        for i in range(n_resnet_blocks):
            self.model.append(ResNetBlock(dim=hidden_dim, activation_function_name=activation_function_name))
        self.model.append(torch.nn.Linear(hidden_dim, output_dim)).append(
            get_activation_model(activation_function_name))

    def forward(self, x: torch.Tensor):
        return self.model(x)


def get_torch_loss(loss_function_name):
    if loss_function_name == 'mse':
        return MSELoss()
    else:
        raise ValueError(F"Unknown loss name {loss_fn}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    ###
    batch_size = 64
    epochs = 10000
    N = None
    input_dim = None
    output_dim = None
    hidden_dim = 64
    n_layers = 5
    activation_function_name = 'relu'
    lr = 1e-3

    dataset = 'toy-relu'
    model = 'nn'
    baseline = None
    loss_function_name = 'mse'
    ###
    if dataset == 'diabetes':
        input_dim = 10
        output_dim = 1
        ds = TorchDiabetesDataset()

    elif dataset == 'toy-linear-1':
        # TODO The nn models works fine with out-dim=1, higher dims (Multi-linear regression, No..) . Dig Deeper
        # TODO, also this data-set works fine with Identity Activation, not with Relu or sigmoid, logical ??
        input_dim = 3
        N = batch_size * 10
        A = torch.Tensor([[0.1, 0.2, -0.3]]).T
        b = torch.Tensor([[1.0]])
        dist = torch.distributions.Normal(loc=1.0, scale=2.0)
        output_dim = 1
        ds = ToyLinearDataSet1(N=N, A=A, b=b, dist=dist)
    elif dataset == 'toy-relu':
        N = batch_size * 10
        input_dim = 10
        output_dim = 1
        ds = ToyRelu(N=N, input_dim=input_dim, out_dim=output_dim)

    else:
        raise ValueError(f'unknown dataset {dataset}')

    # build baseline
    baseline = build_sklearn_diabetes_baseline(X=ds.X.detach().numpy(), y=ds.y.detach().numpy(),
                                               loss_function_name=loss_function_name)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    if model == 'resnet':
        model = ResNetBasic(n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=output_dim, activation_function_name=activation_function_name)
    elif model == 'nn':
        model = NNBasic(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, n_layers=n_layers,
                        activation_function_name=activation_function_name)
    elif model == 'linear':
        model = LinearModel(input_dim=input_dim, out_dim=output_dim)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = get_torch_loss(loss_function_name=loss_function_name)
    rolling_avg_loss = np.inf
    loss_threshold = 1e-2
    epochs_avg_losses = []
    for epoch in tqdm(range(1, epochs + 1)):
        batch_losses = []
        for i, (X, y) in enumerate(dl):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y, y_hat)
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        epochs_avg_losses.append(np.nanmean(batch_losses))
        rolling_avg_loss = np.nanmean(epochs_avg_losses[:-10])
        if epoch % 10 == 0 and epoch > 10:
            logger.info(f'epoch = {epoch}, rolling-avg-loss= {rolling_avg_loss.item()}')
        if rolling_avg_loss < loss_threshold:
            logger.info(f'avg-loss = {rolling_avg_loss}<loss_threshold = {loss_threshold}. Ending the training loop. ')
            break
    # compare to baseline
    print(f'baseline-for dataset {dataset} = {baseline}')
    print(f'Model {model} : avg loss = {np.nanmean(rolling_avg_loss)}')
