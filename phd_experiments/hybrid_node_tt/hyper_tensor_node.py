"""
stability of linear ode
https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2010/2010_4.2%20Stability%20and%20Linearization%20of%20ODEs.pdf
https://www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture19_pde2.pdf
https://faculty.ksu.edu.sa/sites/default/files/stability.pdf
"""
import logging
import random
from typing import Tuple
import numpy as np
import torch.nn
from torch.nn import MSELoss
from torch.utils.data import random_split, DataLoader

from phd_experiments.hybrid_node_tt.basis import Basis
from phd_experiments.hybrid_node_tt.utils import DataSetInstance, get_dataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class HybridTensorNeuralODE(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, basis_params: dict, t_span: Tuple):
        super().__init__()
        self.t_span = t_span
        basis_type = basis_params.get("type", None)
        if not basis_type:
            raise ValueError("basis_type can't be none")
        if basis_type == "poly":
            deg = basis_params.get("deg", None)
            assert deg, "deg cannot be None"
            self.poly_deg = deg
            dims_A = [deg] * input_dim * 2  # +1 to mimic constant term
            dims_Q = [deg] * input_dim + [output_dim]
            # https://en.wikipedia.org/wiki/Matrix_differential_equation
            low = 0.01
            high = 0.05
            self.A = torch.nn.Parameter(torch.distributions.Uniform(low, high).sample(torch.Size(dims_A)))
            self.Q = torch.nn.Parameter(torch.distributions.Uniform(low, high).sample(torch.Size(dims_Q)))
        else:
            raise NotImplementedError(f"Basis type {basis_type} not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Phi = Basis.poly(x=x, t=None, poly_deg=self.poly_deg)
        x=10


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # configs
    N = 4096
    epochs = 20000
    batch_size = 256
    lr = 1e-3
    nn_hidden_dim = 50
    alpha = 1.0
    data_loader_shuffle = False
    # TODO debug boston experiment
    dataset_instance = DataSetInstance.TOY_RELU
    t_span = 0, 1
    train_size_ratio = 0.8
    poly_deg = 4
    # get dataset and loader
    overall_dataset = get_dataset(dataset_instance=dataset_instance, N=N)
    input_dim = overall_dataset.get_input_dim()
    output_dim = overall_dataset.get_output_dim()
    splits = random_split(dataset=overall_dataset, lengths=[train_size_ratio, 1 - train_size_ratio])
    train_dataset = splits[0]
    test_dataset = splits[1]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=data_loader_shuffle)

    # create model
    model = HybridTensorNeuralODE(input_dim=input_dim, output_dim=output_dim,
                                  basis_params={'type': 'poly', 'deg': poly_deg}, t_span=t_span)
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        batches_loss = []
        for i, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            batches_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'At epoch = {epoch} -> avg-batches-loss = {np.nanmean(batches_loss)}')
