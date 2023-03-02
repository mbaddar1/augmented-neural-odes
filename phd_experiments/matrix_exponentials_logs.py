# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html
# https://www.simiode.org/resources/6425/download/5-010-Text-S-MatrixExponential-StudentVersion.pdf
# https://web.mit.edu/18.06/www/Spring17/Matrix-Exponentials.pdf
# https://proofwiki.org/wiki/Decomposition_of_Matrix_Exponential
# https://www.maths.manchester.ac.uk/~higham/talks/ecm12_log.pdf
# https://www.ias.edu/sites/default/files/sns/files/1-matrixlog_tex(1).pdf
# http://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf
# https://arxiv.org/abs/math/0410556
# https://nhigham.com/2020/11/17/what-is-the-matrix-logarithm/
# https://www.mdpi.com/2227-7390/9/17/2018
import scipy.linalg
import torch


def log_taylor_series_mtx(A: torch.Tensor):
    # http://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf eqn 80
    assert len(A.size()) == 2, "A must be a matrix"
    assert A.size()[0] == A.size()[1], " A must be symmetric"
    I = torch.eye(A.size()[0])
    logA_scipy = scipy.linalg.logm(A.detach().numpy())  # as a ref
    norm_A_minus_I = torch.norm(A - I)
    if norm_A_minus_I > 1:
        print(f'warning : norm_A_minus_I = {norm_A_minus_I} > 1')
    M = 1000
    logA = torch.zeros(size=(A.size()[0],A.size(0)))
    #A_minus_I_pow = A-I
    for m in range(1, M + 1):
        logA += float(-1 ^ (m + 1)) / m * torch.pow((A-I),m)
        #A_minus_I_pow = torch.matmul(A_minus_I_pow, A-I)
    err = torch.norm(logA-torch.Tensor(logA_scipy))
    return None


if __name__ == '__main__':
    A = torch.Tensor([[1.5, 0.00], [0.0, 0.5]])
    logA = log_taylor_series_mtx(A)
