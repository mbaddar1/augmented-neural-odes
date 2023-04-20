from random import random
# https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
import random
import numpy as np
import torch.distributions
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader

from phd_experiments.hybrid_node_tt.models import TensorTrainOdeFunc

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def hook(grad):
    pass
class NaiveNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        hidden_dim = 50
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden_dim, out_dim))
        self.net[0].weight.register_hook(hook)

    def forward(self, t, z):
        b = z.size()[0]
        # t_vals = torch.tensor([t]).repeat(b, 1)
        # z_aug = torch.cat([z, t_vals], dim=1)
        return self.net(z)


class TestDataSet1(Dataset):
    def __init__(self, t):
        # y = w.cos(x)+t
        self.N = 1024
        Dx = 2
        # W = torch.tensor([-0.1, 0.2]).type(torch.float32)
        X = torch.distributions.Uniform(10.0, 20.0).sample(torch.Size([self.N, Dx]))
        # X_non_linear = torch.cos(X)
        # Y = torch.einsum('j,bj->b', W, X.type(torch.float32))  # + t
        Y = X[:,0]
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == '__main__':
    epochs = 10000
    batch_size = 32
    Dz = 2
    t = 0.1  # const
    # model = TensorTrainOdeFunc(Dz=Dz, basis_model="poly", unif_low=-0.5, unif_high=0.5, tt_rank=3, poly_deg=3)
    model = NaiveNN(input_dim=Dz, out_dim=1)
    # params = list(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ds = TestDataSet1(t)
    dataloader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    loss_fn = MSELoss()
    for epoch in range(epochs):
        losses = []
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_hat = model.forward(t, X)
            loss = loss_fn(y_hat, y)
            # print(f'epoch = {epoch} - batch = {i} => loss = {loss.item()}')
            loss.backward()
            optimizer.step()
            # print(model.net[0].weight.grad)
            losses.append(loss.item())
        print(f"epoch {epoch} loss = {np.nanmean(losses)}")
        if epoch%100==0:
            pass
        # over_cores_losses = []
        # for core_idx in range(model.A_TT.order):
        #     model.set_learnable_core(core_idx, True)
        #     y_hat = model.forward2(0.1, z=X)
        #     loss_fn = MSELoss()
        #     loss = loss_fn(y,y_hat)
        #     loss.backward()
        #     optimizer.step()
        #     # print(f'epoch = {epoch} - batch = {i} - core = {core_idx} =>{loss.item()}')
        #     model.set_learnable_core(core_idx, False)
        #     optimizer.zero_grad()
        #     over_cores_losses.append(loss.item())
        # print(f'epoch = {epoch} - batch = {i} =>{np.nanmean(over_cores_losses)}')
