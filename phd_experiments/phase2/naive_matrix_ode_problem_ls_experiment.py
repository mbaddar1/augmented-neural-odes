import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch.distributions
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def ode_func(t, z, A):
    z_next = np.matmul(A, z)
    return z_next


def forward(Z0, A, t_span):
    # Dz = Z0.size()[1]
    # Z0_list = list(Z0)
    # # ZT_hat_list = list(
    #     map(lambda z: torch.Tensor(
    #         scipy.integrate.solve_ivp(fun=ode_func, t_span=t_span,
    #                                   y0=z, args=(A.detach().numpy(),)).y[:, -1]).view(1, Dz), Z0_list))
    E = torch.Tensor(scipy.linalg.expm(A.detach().numpy() * (t_span[1] - t_span[0])))

    Z_hat_expm = torch.einsum('ji,bi->bj', E, Z0)
    # ZT_hat = torch.concat(ZT_hat_list)
    # norm_ = torch.norm(Z_hat_expm - ZT_hat)
    return Z_hat_expm


def backward_ls(Z0, ZT, t_span):
    E = torch.linalg.lstsq(Z0, ZT).solution.T
    E_np = E.detach().numpy()
    logE = scipy.linalg.logm(E_np)
    A_ls_np = logE / (t_span[1] - t_span[0])
    A_ls = torch.Tensor(A_ls_np)
    return A_ls


class PolyDataGen(Dataset):
    def __init__(self, M, N, Dx, deg):
        print("Initializing Data-set")
        self.N = N
        unif = torch.distributions.Uniform(low=-10, high=10)
        X = unif.sample(sample_shape=torch.Size([N, Dx]))
        X_pow = torch.pow(X, deg)
        Y = torch.einsum('ji,bi->bj', M, X_pow)

        PolyDataGen.test_expm_reconstruction_error(Y=Y, X=X_pow)
        self.x_train = X_pow
        self.y_train = Y

    @staticmethod
    def test_expm_reconstruction_error(Y, X, t_span=(0, 1)):
        M_ls = torch.linalg.lstsq(X, Y).solution.T
        E = M_ls
        logE = scipy.linalg.logm(E.detach().numpy())
        A_ls = logE / (t_span[1] - t_span[0])
        E2 = scipy.linalg.expm(A_ls * (t_span[1] - t_span[0]))
        Y2 = torch.einsum('ji,bi->bj', torch.Tensor(E2), X)
        mse_reconst_loss = MSELoss()
        loss_reconstr = mse_reconst_loss(Y, Y2)
        assert loss_reconstr.item() < 1e-4

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


if __name__ == '__main__':
    batch_size = 64
    N = 1024
    epochs = 10
    t_span = (0, 0.8)
    Dx = 3
    ###
    M = torch.Tensor([[0.1, 0.8, 0.2], [-0.4, 0.7, 0.1], [0.8, 0.9, -0.3]])
    A_ls_ref = torch.Tensor(scipy.linalg.logm(M.detach().numpy()) / (t_span[1] - t_span[0]))
    ds = PolyDataGen(M, N=N, Dx=Dx, deg=1)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    A = torch.distributions.Uniform(low=0.01, high=0.1).sample(sample_shape=torch.Size([Dx, Dx]))
    mse_loss = MSELoss()
    alpha = 0.8
    losses = []
    for epoch in tqdm(range(epochs), desc='epochs'):
        for i, (Z0, ZT) in enumerate(loader):
            E_ls = torch.linalg.lstsq(Z0,ZT).solution.T
            logE_ls = scipy.linalg.logm(E_ls.detach().numpy())
            A_ls_2 = logE_ls/(t_span[1]-t_span[0])
            E_ls_fw = torch.Tensor(scipy.linalg.expm(A_ls_2*(t_span[1]-t_span[0])))
            Z_hat_2 = torch.einsum('ji,bi->bj',E_ls_fw,Z0)
            loss2 = mse_loss(Z_hat_2,ZT)
            #####
            # ZT_hat = forward(Z0=Z0, A=A, t_span=t_span)
            # Z_hat_ref = forward(Z0=Z0, A=A_ls_ref, t_span=t_span)
            # norm1 = torch.norm(ZT_hat - Z_hat_ref)
            # loss = mse_loss(ZT, ZT_hat)
            # A_ls = backward_ls(Z0=Z0, ZT=ZT, t_span=t_span)
            # norm_ = torch.norm(A_ls - A_ls_ref)
            # A = alpha * A_ls + (1 - alpha) * A
            # norm2_ = torch.norm(A - A_ls_ref)
            losses.append(loss2.item())
    print("-----------------------------------")
    print('####### Losses ################')
    print(losses)
    plt.plot(losses)
    plt.savefig('losses.png')
    print('#################################')
