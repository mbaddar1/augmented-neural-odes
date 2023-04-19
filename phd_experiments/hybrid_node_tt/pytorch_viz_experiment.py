import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace

from phd_experiments.hybrid_node_tt.models import TensorTrainFixedRank

# model = nn.Sequential()
# model.add_module('W0', nn.Linear(8, 16))
# model.add_module('tanh', nn.Tanh())
# model.add_module('W1', nn.Linear(16, 1))
#
# x = torch.randn(1,8)
#
# make_dot(model(x), params=dict(model.named_parameters())).render("attached", format="png")
deg = 3
rank = 6
Dx = 5
b = 128
order = Dx + 1  # +1 for time
ttxr = TensorTrainFixedRank(dims=[deg + 1] * (Dx + 1), fixed_rank=3, requires_grad=True, unif_low=-0.01, unif_high=0.01,
                            poly_deg=deg)
param_list1 = list(ttxr.named_parameters())
param_list2 = list(ttxr.named_parameters())

x = torch.distributions.Uniform(0.01, 2).sample(sample_shape=torch.Size([b, Dx]))
i = 3
t = 0.2
for i in range(order):
    ttxr.set_learnable_core(i=i, req_grad=True)
    y = ttxr.forward2(t=t, z=x)
    ttxr.set_learnable_core(i=i, req_grad=False)
    make_dot(y, params=dict(ttxr.named_parameters())).render(f"ttxr_{i}", format="png")

