"""
this script is to implement high-dim, tensor-exponent
"""
import logging
import string
from typing import List

import numpy as np
from scipy.linalg import expm
import torch

from phd_experiments.hybrid_node_tt.tensor_power import TensorPower
from phd_experiments.hybrid_node_tt.utils import generate_einsum_string, prod_list, generate_identity_tensor, \
    assert_dims_symmetry


# Square tensors
# https://www.sciencedirect.com/science/article/pii/S0898122118300798
# https://www.tandfonline.com/doi/full/10.1080/03081087.2015.1071311
# Drazin inverse https://en.wikipedia.org/wiki/Drazin_inverse
# Drazin inverse applied to square tensors
# https://www.sciencedirect.com/science/article/pii/S0898122118300798 (p2)


class TensorExponent:
    TOL = 1e-6

    @staticmethod
    def calculate(A: torch.Tensor, method="power", **kwargs):
        """

        """
        """
        A => C^{I1 x I2 x ... x Ik x I1 x I2 x .... x Ik} (even-order square Tensor)
        
        """
        dims = list(A.size())
        assert_dims_symmetry(dims), f"symmetry assertion failed , dims = {dims}"
        if method == "power":
            power_max = kwargs.get("power_max", None)  # if power_max is none , adaptive
            return TensorExponent._tensor_exponent_power(A=A, power_max=power_max)

    """
    # TODO Truncation error for Taylor Series
    # make stop criteria either some max power or based on truncation error
    # https://www.youtube.com/watch?v=Cqi-b3nQdKM
    # https://www.efunda.com/math/taylor_series/exponential.cfm
    # https://youtu.behttps://web.engr.oregonstate.edu/~webbky/MAE4020_5020_files/Section%204%20Roundoff%20and%20Truncation%20Error.pdf/ywk2HpTL0W0
    # https://brilliant.org/wiki/taylor-series-error-bounds/
    # https://www.youtube.com/watch?v=YgJSP9DBsDI
    # https://math.stackexchange.com/questions/67422/how-to-bound-the-truncation-error-for-a-taylor-polynomial-approximation-of-tan
    """

    # power series method
    @staticmethod
    def _tensor_exponent_power(A: torch.Tensor, power_max):
        dims = list(A.size())
        I = generate_identity_tensor(dims=dims)
        exp_acc = I
        factorial_term = 1
        for power_ in range(1, power_max + 1):
            factorial_term *= 1.0 / power_
            A_exp, _ = TensorPower.calculate(A, power_)
            term = factorial_term * A_exp
            if power_ > 10:  # FIXME : to add some adaptivity, make it smarter
                if torch.norm(term) < TensorExponent.TOL:
                    print(f'breaking taylor series at power = {power_}')
                    break
            exp_acc+=term
        return exp_acc


if __name__ == '__main__':
    dims = [2, 4, 2, 4]
    order = len(dims)
    k = int(order / 2)
    mtx_dims = [prod_list(dims[:k]), prod_list(dims[k:])]
    assert_dims_symmetry(dims), "dims symmetry check failed"
    A = torch.distributions.Uniform(0, 1).sample(torch.Size(dims)).type(torch.float64)
    my_expA = TensorExponent.calculate(A=A, power_max=30, verify=True)
    expA_truth = torch.reshape(torch.tensor(expm(torch.reshape(A, mtx_dims).detach().numpy())), dims)
    err_norm = torch.norm(my_expA - expA_truth)
    print(err_norm.item())
