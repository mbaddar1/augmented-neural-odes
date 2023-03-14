import torch
from torch.nn import MSELoss


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


def get_torch_loss(loss_function_name):
    if loss_function_name == 'mse':
        return MSELoss()
    else:
        raise ValueError(F"Unknown loss name {loss_function_name}")
