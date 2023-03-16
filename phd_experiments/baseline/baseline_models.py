"""
From scratch impl.
https://d2l.ai/chapter_convolutional-modern/resnet.html
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
"""
import logging
import random

import numpy as np
import torch.nn
import torchmetrics
from sklearn import linear_model
from torch.nn import Sequential
from torch.utils.data import DataLoader
from tqdm import tqdm

from phd_experiments.datasets.torch_boston_housing import TorchBostonHousingPrices
from phd_experiments.datasets.torch_sklearn_diabetes import TorchDiabetesDataset
from phd_experiments.datasets.toy_linear import ToyLinearDataSet1
from phd_experiments.datasets.toy_relu import ToyRelu
from phd_experiments.utils.torch_utils import get_activation_model, get_torch_loss
from sklearn.metrics import r2_score

# Reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
# https://sklearn-genetic-opt.readthedocs.io/en/stable/tutorials/reproducibility.html
SEED = 54
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def build_sklearn_linear_regression_baseline(X: np.ndarray, y: np.ndarray, loss_function_name: str):
    # https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    return regr


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

    dataset_name = 'boston'
    model = 'resnet'
    naive_baseline_model = None
    loss_function_name = 'mse'
    ###
    if dataset_name == 'diabetes':

        input_dim = 10
        output_dim = 1
        overall_dataset = TorchDiabetesDataset()

    elif dataset_name == 'toy-linear-1':
        # TODO The nn models works fine with out-dim=1, higher dims (Multi-linear regression, No..) . Dig Deeper
        # TODO, also this data-set works fine with Identity Activation, not with Relu or sigmoid, logical ??
        input_dim = 3
        N = batch_size * 10
        A = torch.Tensor([[0.1, 0.2, -0.3]]).T
        b = torch.Tensor([[1.0]])
        dist = torch.distributions.Normal(loc=1.0, scale=2.0)
        output_dim = 1
        overall_dataset = ToyLinearDataSet1(N=N, A=A, b=b, dist=dist)
    elif dataset_name == 'toy-relu':
        N = batch_size * 10
        input_dim = 10
        output_dim = 1
        overall_dataset = ToyRelu(N=N, input_dim=input_dim, out_dim=output_dim)
    elif dataset_name == 'boston':
        csv_file = '../datasets/boston.csv'
        overall_dataset = TorchBostonHousingPrices(csv_file=csv_file)
        input_dim = overall_dataset.get_Xdim()
        output_dim = 1

    else:
        raise ValueError(f'unknown dataset {dataset_name}')

    # build baseline
    naive_baseline_model = build_sklearn_linear_regression_baseline(X=overall_dataset.X.detach().numpy(),
                                                                    y=overall_dataset.y.detach().numpy(),
                                                                    loss_function_name=loss_function_name)
    # generate datasets
    datasets = torch.utils.data.random_split(dataset=overall_dataset, lengths=[0.8, 0.2],
                                             generator=torch.Generator().manual_seed(42))
    train_dataset = datasets[0]
    test_dataset = datasets[1]
    dataloader = DataLoader(dataset=overall_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    # select model
    if model == 'resnet':
        model = ResNetBasic(n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                            output_dim=output_dim, activation_function_name=activation_function_name)
    elif model == 'nn':
        model = NNBasic(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=output_dim, n_layers=n_layers,
                        activation_function_name=activation_function_name)
    elif model == 'linear':
        model = LinearModel(input_dim=input_dim, out_dim=output_dim)
    # train model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_fn = get_torch_loss(loss_function_name=loss_function_name)
    rolling_avg_loss = np.inf
    loss_threshold = 1e-2
    epochs_avg_losses = []
    for epoch in tqdm(range(1, epochs + 1)):
        batch_losses = []
        for i, (X, y) in enumerate(train_dataloader):
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
    # evaluate model vs baseline
    r2Score = torchmetrics.R2Score()
    r2score_model_list = []
    r2score_naive_baseline_list = []
    for j, (X, y) in enumerate(test_dataloader):
        y_pred_model = model(X)
        y_pred_baseline = naive_baseline_model.predict(X.detach().numpy())
        r2score_val_model = r2Score(y_pred_model, y)
        r2score_naive_baseline = r2_score(y_true=y.detach().numpy(), y_pred=y_pred_baseline)
        r2score_naive_baseline_list.append(r2score_val_model.detach().numpy())
        r2score_model_list.append(r2score_naive_baseline)
    logger.info('Out of sample evaluation summary\n==============================================\n')
    logger.info(f'Dataset = {dataset_name}')
    logger.info(f'Baseline-Model = {naive_baseline_model}')
    logger.info(f'Model = {model}')
    logger.info(
        f'Model r2score = {np.nanmean(r2score_model_list)} <=> Baseline r2score {np.nanmean(r2score_naive_baseline_list)}')
    logger.info('=============================================================')
