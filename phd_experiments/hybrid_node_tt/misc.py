import torch

from phd_experiments.hybrid_node_tt.torch_rbf import RBF, basis_func_dict

if __name__ == '__main__':
    in_features = 2
    n_centers = 3 # out_features for rbf
    N = 100
    X = torch.distributions.Normal(0,1).sample(torch.Size([N,in_features]))
    basis_fn = basis_func_dict()["gaussian"]
    rbf_model = RBF(in_features=in_features, n_centres=n_centers, basis_func=basis_fn)
    out = rbf_model(X)

