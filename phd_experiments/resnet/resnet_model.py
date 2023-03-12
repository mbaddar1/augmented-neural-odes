"""
From scratch impl.
https://d2l.ai/chapter_convolutional-modern/resnet.html
https://debuggercafe.com/implementing-resnet18-in-pytorch-from-scratch/
https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
"""
import torch.nn
from torch.nn import Sequential


def get_activation_model(activation_function_name: str):
    if activation_function_name == 'relu':
        return torch.nn.Module()
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

    def __init__(self, input_dim, hidden_dim, out_dim, activation_function_name):
        super().__init__()

        activation_model = get_activation_model(activation_function_name)

        self.model = Sequential(torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
                                activation_model,
                                torch.nn.Linear(in_features=hidden_dim, out_features=out_dim),
                                activation_model)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class ResNetBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, activation_function_name):
        # https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
        # https://d2l.ai/_images/residual-block.svg
        super().__init__()
        self.non_linear_block = ResNetNonLinearSubBlock(input_dim=input_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                                                        activation_function_name=activation_function_name)

    def forward(self, x):
        return x + self.non_linear_block(x)  # f(x) = g(x) + x


class ResNetBasic(torch.nn.Module):
    """
    https://d2l.ai/chapter_convolutional-modern/resnet.html#resnet-model
    https://d2l.ai/_images/resnet18-90.svg
    """

    def __init__(self, n_layers: int, hidden_dim: int, activation_function: str):
        super().__init__()
        pass
