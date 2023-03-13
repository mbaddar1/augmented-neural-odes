"""
From scratch impl.
https://d2l.ai/chapter_convolutional-modern/resnet.html
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
"""
import numpy as np
import sklearn.model_selection
import torch.nn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader

from phd_experiments.datasets.torch_sklearn_diabetes import TorchDiabetesDataset
from phd_experiments.datasets.toy_linear import ToyLinearDataSet1
from phd_experiments.datasets.toy_relu import ToyRelu


def build_sklearn_diabetes_baseline():
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    epochs = 100
    mse_list = []
    for i in range(epochs):
        X, y = datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
    ret = {'model': str(regr.__class__), 'avg-mse': np.nanmean(mse_list), 'std-mse': np.nanstd(mse_list)}
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


if __name__ == '__main__':

    batch_size = 32
    epochs = 1000
    N = None
    input_dim = None
    output_dim = None
    hidden_dim = 64
    n_layers = 5
    activation_function_name = 'identity'
    lr = 0.1
    mse_loss_fn = MSELoss()
    dataset = 'diabetes'
    model = 'linear'
    baselines = {}
    ###
    if dataset == 'diabetes':
        input_dim = 10
        output_dim = 1
        ds = TorchDiabetesDataset()
        ret = build_sklearn_diabetes_baseline()
        print(f'diabetes baseline model (linear-regression) score = {ret}')

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

    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    if model == 'baseline':
        model = ResNetBasic(n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=output_dim, activation_function_name=activation_function_name)
    elif model == 'nn':
        model = NNBasic(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, n_layers=n_layers,
                        activation_function_name=activation_function_name)
    elif model == 'linear':
        model = LinearModel(input_dim=input_dim, out_dim=output_dim)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        for i, (X, y) in enumerate(dl):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = mse_loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'epoch = {epoch}, loss= {loss.item()}')
