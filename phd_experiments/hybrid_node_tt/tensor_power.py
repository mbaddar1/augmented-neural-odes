"""
https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/
https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L546-L659
https://www.hackerearth.com/practice/notes/matrix-exponentiation-1/
"""
import torch


class TensorPower:
    @staticmethod
    def calculate(A: torch.Tensor, n: int) -> torch.Tensor:
        dims = list(A.size())
        assert len(dims) == 2
        bin_n = format(n, 'b')
        bin_n = bin_n[::-1]
        # len_bin_b = len(bin_n)
        A_power_final = torch.eye(dims[0])
        A_power_prod = None
        for i, e in enumerate(bin_n):
            if i == 0:
                A_power_prod = A
            else:
                A_power_prod = torch.einsum('ij,jk->ik', A_power_prod, A_power_prod)
            if int(e) == 1:
                A_power_final = torch.einsum('ij,jk->ik', A_power_final, A_power_prod)
        return A_power_final


if __name__ == '__main__':
    A = torch.tensor([[0.1, 0.3], [0.4, 0.6]])
    n = 20
    A_my_pow = TensorPower.calculate(A, n)

    A_pow_torch = torch.linalg.matrix_power(A,n)
    err = torch.norm(A_my_pow-A_pow_torch)
    print(err.item())
