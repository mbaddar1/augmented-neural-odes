import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch.distributions
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45


def ode_func(t, z, A):
    if isinstance(A, np.ndarray) and isinstance(z, np.ndarray):
        z_next = np.matmul(A, z)
    elif isinstance(A, torch.Tensor) and isinstance(z, torch.Tensor):
        z_next = torch.einsum('ji,bi->bj', A, z)
    else:
        raise ValueError("type of A or z is not valid")
    return z_next


def forward(Z0, A, t_span):
    E = torch.Tensor(scipy.linalg.expm(A.detach().numpy() * (t_span[1] - t_span[0])))
    Z_hat_expm = torch.einsum('ji,bi->bj', E, Z0)
    return Z_hat_expm


def backward_ls(Z0, ZT, t_span):
    E = torch.linalg.lstsq(Z0, ZT).solution.T
    E_np = E.detach().numpy()
    logE = scipy.linalg.logm(E_np)
    A_ls_np = logE / (t_span[1] - t_span[0])
    A_ls = torch.Tensor(A_ls_np)
    return A_ls


def backward_trajectory_ls(Z0, ZT):
    pass


class PolyDataGen(Dataset):
    def __init__(self, M, N, Dx, deg):
        print("Initializing Data-set")
        self.noise_scale = 0.1
        self.N = N
        unif = torch.distributions.Uniform(low=-10, high=10)
        X = unif.sample(sample_shape=torch.Size([N, Dx]))
        Phi = torch.pow(X, deg)
        Y = torch.einsum('ji,bi->bj', M, Phi)
        noise_dist = torch.distributions.Normal(loc=0, scale=self.noise_scale)
        Y += noise_dist.sample(Y.size())
        PolyDataGen.test_expm_reconstruction_error(Y=Y, Phi=Phi, noise_scale=self.noise_scale)
        self.x_train = X
        self.y_train = Y

    @staticmethod
    def test_expm_reconstruction_error(Y, Phi, noise_scale, t_span=(0, 1)):
        M_ls = torch.linalg.lstsq(Phi, Y).solution.T
        E = M_ls
        logE = scipy.linalg.logm(E.detach().numpy())
        A_ls = logE / (t_span[1] - t_span[0])
        E2 = scipy.linalg.expm(A_ls * (t_span[1] - t_span[0]))
        Y2 = torch.einsum('ji,bi->bj', torch.Tensor(E2), Phi)

        mse_reconst_loss = MSELoss()
        loss_reconstr = mse_reconst_loss(Y, Y2)
        assert loss_reconstr.item() < 2 * noise_scale, f"Reconstruction mse = {loss_reconstr.item()} "

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


if __name__ == '__main__':
    batch_size = 128
    N = 1024
    epochs = 100
    t_span = (0, 1.0)
    Dx = 3
    deg = 5
    ###
    M = torch.Tensor([[0.1, 0.8, 0.2], [-0.4, 0.7, 0.1], [0.8, 0.9, -0.3]])
    A_ls_ref = torch.Tensor(scipy.linalg.logm(M.detach().numpy()) / (t_span[1] - t_span[0]))
    ds = PolyDataGen(M, N=N, Dx=Dx, deg=deg)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    # A = torch.distributions.Uniform(low=0.01, high=0.1).sample(sample_shape=torch.Size([Dx, Dx]))
    mse_loss = MSELoss()
    alpha = 0.2
    losses = []
    A_ls = None
    A = np.random.uniform(low=0.01, high=0.1, size=[Dx, Dx])
    for epoch in tqdm(range(epochs), desc='epochs'):
        epoch_losses = []
        for i, (Z0, ZT) in enumerate(loader):
            # forward
            # basis fn
            Phi = torch.pow(Z0, exponent=deg)
            # forward via closed form
            E_ls_fw = torch.Tensor(scipy.linalg.expm(A * (t_span[1] - t_span[0])))
            Z_hat_expm = torch.einsum('ji,bi->bj', E_ls_fw, Phi)
            solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=Z0.dtype)
            Z_hat_integ = solver.solve_ivp(func=ode_func, t_span=t_span, z0=Z0, args=(torch.Tensor(A),))
            # forward via integration

            # loss computation
            loss = mse_loss(Z_hat_expm, ZT)
            epoch_losses.append(loss.item())

            # backward ls
            # i) closed form
            E_ls = torch.linalg.lstsq(Phi, ZT).solution.T.detach().numpy()
            logE_ls = scipy.linalg.logm(E_ls)
            A_ls_closed_form = logE_ls / (t_span[1] - t_span[0])

            # update
            A = alpha * A_ls_closed_form + (1 - alpha) * A
        losses.append(np.nanmean(epoch_losses))

    print("-----------------------------------")
    print('####### Losses ################')
    print(losses)
    plt.plot(losses)
    plt.savefig('losses.png')
    print('#################################')
