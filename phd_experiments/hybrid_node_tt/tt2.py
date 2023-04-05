import logging
from typing import List

import torch
from torch import Tensor

from phd_experiments.hybrid_node_tt.basis import Basis


class TensorTrainFixedRank(torch.nn.Module):
    def __init__(self, dims: List[int], fixed_rank: int, requires_grad: bool, unif_low: float, unif_high: float):
        """
        dims : dims[0]-> output dim
        dims[1:order] -> input dims
        """
        super().__init__()
        order = len(dims)
        tensor_uniform = torch.distributions.Uniform(low=unif_low, high=unif_high)
        # https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html
        self.core_tensors = torch.nn.ParameterDict(
            {'G0': torch.nn.Parameter(tensor_uniform.sample(torch.Size([dims[1], fixed_rank])))})
        self.out_dim = dims[0]
        for i in range(2, order - 1):
            self.core_tensors[f"G{i - 1}"] = torch.nn.Parameter(
                tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[i], fixed_rank])))

        self.core_tensors[f"G{order - 2}"] = torch.nn.Parameter(
            tensor_uniform.sample(sample_shape=torch.Size([fixed_rank, dims[order - 1], self.out_dim])),
            requires_grad=requires_grad)
        assert len(self.core_tensors) == order - 1, \
            f"# core tensors should == order-1 : {len(self.core_tensors)} != {order - 1}"

    def norm(self):
        tot_norm = 0

        for tensor in self.core_tensors.values():
            tot_norm += torch.norm(tensor)
        return tot_norm

    def display_sizes(self):
        sizes = []
        for tensor in self.core_tensors:
            sizes.append(tensor.size())
        self.logger.info(f'TensorTrain sizes = {sizes}')

    def is_parameter(self):
        requires_grads = list(map(lambda x: x.requires_grad, self.core_tensors))
        return all(requires_grads)

    def forward(self, t: float, z: Tensor, poly_deg: int) -> Tensor:
        Phi = Basis.poly(z=z, t=t, poly_deg=poly_deg)
        assert len(self.core_tensors) == len(
            Phi), f"# of core-tensors must == number of basis tensors " \
                  f", {len(self.core_tensors)} != {len(Phi)}"
        n_cores = len(self.core_tensors)
        # first core
        res_tensor = torch.einsum("ij,bi->bj", self.core_tensors[f"G{0}"], Phi[0])
        # middle cores
        for i in range(1, len(self.core_tensors) - 1):
            core_basis = torch.einsum("ijk,bj->bik", self.core_tensors[f"G{i}"], Phi[i])
            res_tensor = torch.einsum("bi,bik->bk", res_tensor, core_basis)
        # last core
        core_basis = torch.einsum("ijl,bj->bil", self.core_tensors[f"G{n_cores - 1}"], Phi[n_cores - 1])
        res_tensor = torch.einsum("bi,bil->bl", res_tensor, core_basis)
        assert res_tensor.size()[1] == self.out_dim, f"output tensor size must = " \
                                                     f"out_dim : {res_tensor.size()}!={self.out_dim}"
        return res_tensor

    def display_cores(self):
        self.logger.info(f'Cores : \n{self.core_tensors}\n')


if __name__ == '__main__':
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    main_logger = logging.getLogger()
    order = 4
    basis_dim = 3
    out_dim = 10
    fixed_rank = 2

    ttxr = TensorTrainFixedRank(order=order, core_input_dim=basis_dim, out_dim=out_dim, fixed_rank=fixed_rank,
                                requires_grad=True)
    norm_val = ttxr.norm()
    ttxr.display_sizes()
    ttxr.is_parameter()
    main_logger.info(ttxr.is_parameter())

    basis_ = []
    for i in range(order):
        basis_.append(torch.ones(basis_dim))
    res = ttxr.contract_basis(basis_tensors=basis_)
