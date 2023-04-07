"""
https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/
https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L546-L659
https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/
"""
from typing import Tuple

import torch

from phd_experiments.hybrid_node_tt.utils import generate_einsum_string_tensor_power, generate_identity_tensor, assert_dims_symmetry, \
    prod_list


class TensorPower:
    @staticmethod
    def calculate(A: torch.Tensor, n: int) -> Tuple[torch.Tensor, int]:
        dims = list(A.size())
        order = len(dims)
        # FIXME : don't generate everytime, generate once
        einsum_str = generate_einsum_string_tensor_power(order)
        contract_op_count = 0
        bin_n = format(n, 'b')
        bin_n = bin_n[::-1]
        A_power_final = generate_identity_tensor(dims=dims)
        A_power_prod = None
        for i, e in enumerate(bin_n):
            if i == 0:
                A_power_prod = A
            else:
                A_power_prod = torch.einsum(einsum_str, A_power_prod, A_power_prod)
                contract_op_count += 1
            if int(e) == 1:
                A_power_final = torch.einsum(einsum_str, A_power_final, A_power_prod)
                contract_op_count += 1
        return A_power_final, contract_op_count


if __name__ == '__main__':
    dims = [4, 4]
    assert_dims_symmetry(dims=dims), "dims symmetry check failed"
    order = len(dims)
    k = int(order / 2)
    mtx_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
    A = torch.distributions.Uniform(0.1, 0.2).sample(torch.Size(dims)).type(torch.float64)

    A_mtx_torch = torch.reshape(A, mtx_dims)
    n = 10
    A_my_pow, count_ = TensorPower.calculate(A, n)
    A_pow_truth_mtx = torch.linalg.matrix_power(A_mtx_torch, n)
    A_pow_truth_tensor = torch.reshape(A_pow_truth_mtx, dims)
    err = torch.norm(A_my_pow - A_pow_truth_tensor)
    print(err.item())
    print(count_)
