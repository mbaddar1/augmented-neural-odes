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
    Dz = Z0.size()[1]
    Z0_list = list(Z0)
    ZT_hat_list = list(
        map(lambda z: torch.Tensor(
            scipy.integrate.solve_ivp(fun=ode_func, t_span=t_span,
                                      y0=z, args=(A.detach().numpy(),)).y[:, -1]).view(1, Dz), Z0_list))
    ZT_hat = torch.concat(ZT_hat_list)
    return ZT_hat


def backward_ls(Z0, ZT, t_span):
    E = torch.linalg.lstsq(Z0, ZT).solution.T
    E_np = E.detach().numpy()
    logE = scipy.linalg.logm(E_np)
    A_ls_np = logE / (t_span[1] - t_span[0])
    A_ls = torch.Tensor(A_ls_np)
    return A_ls


class SynMtxOdeDataSet(Dataset):
    def __init__(self, N, Dz, t_span):
        print("Initializing Data-set")
        self.N = N
        unif = torch.distributions.Uniform(low=-10, high=10)
        Z0 = unif.sample(sample_shape=torch.Size([N, Dz]))
        A = 1e-3 * torch.Tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]])
        ZT = forward(Z0=Z0, A=A, t_span=t_span)
        self.x_train = Z0
        self.y_train = ZT

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


if __name__ == '__main__':
    batch_size = 64
    N = 2048
    epochs = 100
    t_span = (0, 0.5)
    Dz = 3
    ###
    ds = SynMtxOdeDataSet(N=N, Dz=Dz, t_span=t_span)
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    A = torch.distributions.Uniform(low=0.01, high=0.1).sample(sample_shape=torch.Size([Dz, Dz]))
    mse_loss = MSELoss()
    alpha = 0.8
    losses = []
    for epoch in tqdm(range(epochs), desc='epochs'):
        for i, (Z0, ZT) in enumerate(loader):
            ZT_hat = forward(Z0=Z0, A=A, t_span=t_span)
            loss = mse_loss(ZT, ZT_hat)
            A_ls = backward_ls(Z0=Z0, ZT=ZT, t_span=t_span)
            A = alpha * A_ls + (1 - alpha) * A
            losses.append(loss.item())
    print("-----------------------------------")
    print('####### Losses ################')
    print(losses)
    print('#################################')
