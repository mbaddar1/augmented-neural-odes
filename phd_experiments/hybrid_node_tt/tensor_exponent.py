"""
this script is to implement high-dim, tensor-exponent
"""
import logging
import string
from typing import List

import numpy as np
import scipy.linalg
import torch

from phd_experiments.hybrid_node_tt.utils import generate_einsum_string


# Square tensors
# https://www.sciencedirect.com/science/article/pii/S0898122118300798
# https://www.tandfonline.com/doi/full/10.1080/03081087.2015.1071311
# Drazin inverse https://en.wikipedia.org/wiki/Drazin_inverse
# Drazin inverse applied to square tensors
# https://www.sciencedirect.com/science/article/pii/S0898122118300798 (p2)

def prod_list(l: List):
    prod_ = 1
    for e in l:
        prod_ *= e
    return prod_


class TensorExponent:
    @staticmethod
    def calculate(A: torch.Tensor, method="power", verify=False, **kwargs):
        """

        """
        """
        A => C^{I1 x I2 x ... x Ik x I1 x I2 x .... x Ik} (even-order square Tensor)
        
        """
        dims = list(A.size())
        TensorExponent._assert_dims_symmetry(dims), f"symmetry assertion failed , dims = {dims}"
        if method == "power":
            power_max = kwargs.get("power_max", None)  # if power_max is none , adaptive
            TensorExponent._tensor_exponent_power(A=A, power_max=power_max, verify=verify)

    @staticmethod
    def _assert_dims_symmetry(dims: List[int]):
        order = len(dims)
        assert order > 0 and order % 2 == 0, f"order must be even and > 0 , got order = {order}"
        k = int(order / 2)
        for i in range(k):
            assert dims[i] == dims[i + k], f"dims[{i + 1}] = {dims[i]} != dims{[i + k]} = {dims[i + k]}"
        return True

    # power series method
    @staticmethod
    def _tensor_exponent_power(A: torch.Tensor, power_max, verify: bool):
        dims = list(A.size())
        order = len(dims)
        k = int(order / 2)
        einsum_str = generate_einsum_string(len(A.size()))
        if verify:
            mtx_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
            A_mtx_np = torch.reshape(A, mtx_dims).detach().numpy()

        # TODO Truncation error for Taylor Series
        #   make stop criteria either some max power or based on truncation error
        # https://www.youtube.com/watch?v=Cqi-b3nQdKM
        # https://www.efunda.com/math/taylor_series/exponential.cfm
        # https://youtu.behttps://web.engr.oregonstate.edu/~webbky/MAE4020_5020_files/Section%204%20Roundoff%20and%20Truncation%20Error.pdf/ywk2HpTL0W0
        # https://brilliant.org/wiki/taylor-series-error-bounds/
        # https://www.youtube.com/watch?v=YgJSP9DBsDI
        # https://math.stackexchange.com/questions/67422/how-to-bound-the-truncation-error-for-a-taylor-polynomial-approximation-of-tan

        I = TensorExponent.generate_identity_tensor(dims=dims, verify=verify)
        A_exp_prod = I
        exp_acc = I
        factorial_term = 1
        for power_ in range(1, power_max + 1):
            factorial_term *= 1.0 / power_
            A_exp_prod = torch.einsum(einsum_str, A_exp_prod, A)
            if verify:
                A_mtx_pow_np = np.linalg.matrix_power(A_mtx_np, power_)
                A_exp_prod_mtx_np = torch.reshape(A_exp_prod, mtx_dims)
                n = prod_list(dims)
                err_norm = torch.norm(torch.tensor(A_mtx_pow_np) - (A_exp_prod_mtx_np)).item() / float(n)
                assert err_norm < 0.01, f"A^{power_} failed"
            exp_acc += factorial_term * A_exp_prod
        if verify:
            mtx_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
            A_mtx = torch.reshape(A, mtx_dims).detach().numpy()
            exp_A_mtx = scipy.linalg.expm(A_mtx)
            exp_A_tensor = torch.reshape(torch.tensor(exp_A_mtx), dims)
            n = prod_list(dims)
            err_norm = torch.norm(exp_A_tensor - exp_acc) / float(n)
            assert err_norm < 1e-3, f"large error-norm in tensor-exp = {err_norm.item()}"
        return exp_acc

    @staticmethod
    def generate_identity_tensor(dims: List[int], verify=False):
        assert TensorExponent._assert_dims_symmetry(dims), "Symmetry assertion failed"
        order = len(dims)
        k = int(order / 2)
        matricized_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
        assert matricized_dims[0] == matricized_dims[1], f"matricized dims are not symmetric : {matricized_dims}"
        matrix_identity = torch.eye(matricized_dims[0])
        identity_tensor = torch.reshape(matrix_identity, dims)
        dummy_tensor = torch.distributions.Uniform(0, 1).sample(dims)
        if verify:
            einsum_str = generate_einsum_string(order)
            res = torch.einsum(einsum_str, dummy_tensor, identity_tensor)
            err_norm = torch.norm(res - dummy_tensor)
            assert err_norm.item() < 1e-3, "Identity tensor is wrong"
        return identity_tensor


if __name__ == '__main__':
    torch.linalg.matrix_power(torch.tensor([[2, 2], [2, 2]]), 4)
    dims = [4, 4, 4, 4]
    A = torch.distributions.Uniform(0, 1).sample(torch.Size(dims))
    # eA = TensorExponent.calculate(A)
    TensorExponent.generate_identity_tensor(dims, verify=True)
    TensorExponent.calculate(A=A, power_max=10, verify=True)
