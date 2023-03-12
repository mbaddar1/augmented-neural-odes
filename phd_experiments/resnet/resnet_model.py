"""
From scratch impl.
https://d2l.ai/chapter_convolutional-modern/resnet.html
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
"""
import torch.nn
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader

from phd_experiments.datasets.torch_diabetes import TorchDiabetesDataset


def get_activation_model(activation_function_name: str):
    if activation_function_name == 'relu':
        return torch.nn.ReLU()
    if activation_function_name == 'tanh':
        return torch.nn.Tanh()
    elif activation_function_name == 'sigmoid':
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f'Unknown activation name = {activation_function_name}')


class ResNetNonLinearSubBlock(torch.nn.Module):
    """
    for residual block f(x) = g(x)+x , this class implements g(x)
    """

    def __init__(self, dim, activation_function_name):
        super().__init__()

        activation_model = get_activation_model(activation_function_name)

        self.model = Sequential(torch.nn.Linear(in_features=dim, out_features=dim),
                                activation_model,
                                torch.nn.Linear(in_features=dim, out_features=dim),
                                activation_model)

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

    def __init__(self, n_layers: int, input_dim: int, hidden_dim: int, output_dim: int, activation_function_name: str):
        super().__init__()
        # a bit of modification on the arch in https://d2l.ai/chapter_convolutional-modern/resnet.html#resnet-model
        activation_model = get_activation_model(activation_function_name=activation_function_name)
        self.model = Sequential()
        # add init model
        self.model.append(torch.nn.Linear(input_dim, hidden_dim)).append(activation_model)
        # add resnet blocks
        assert n_layers >= 3, "Number of layers must be >=3"
        n_resnet_blocks = n_layers - 2
        for i in range(n_resnet_blocks):
            self.model.append(ResNetBlock(dim=hidden_dim,activation_function_name=activation_function_name))
        self.model.append(torch.nn.Linear(hidden_dim, output_dim)).append(activation_model)

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == '__main__':
    batch_size = 64
    epochs = 1000
    input_dim = 10
    output_dim = 1
    hidden_dim = 64
    n_layers = 3
    activation_function_name = 'relu'
    lr = 0.1
    mse_loss_fn = MSELoss()
    ###
    ds = TorchDiabetesDataset()
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    res_net_basic_model = ResNetBasic(n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                                      output_dim=output_dim, activation_function_name=activation_function_name)
    optimizer = torch.optim.Adam(params=res_net_basic_model.parameters(), lr=lr)
    for epoch in range(1,epochs+1):
        for i, (X, y) in enumerate(dl):
            optimizer.zero_grad()
            y_hat = res_net_basic_model(X)
            loss = mse_loss_fn(y, y_hat)
            loss.backward()
            optimizer.step()
        if epoch%10==0:
            print(f'epoch = {epoch}, loss= {loss.item()}')
