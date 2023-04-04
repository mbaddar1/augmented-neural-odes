import torch

if __name__ == '__main__':
    # S = torch.tensor([[1.0, 2], [-1, 4]])
    # Lambda = torch.diag(input=torch.tensor([-0.1, -0.2]))
    # A = torch.matmul(S, Lambda)
    # A = torch.matmul(A, torch.inverse(S))
    # e1 = torch.linalg.eigvals(A)
    # print(e1)
    # b = 64
    # deg = 3
    # order = 4
    # A = torch.distributions.Uniform(0, 1).sample(torch.Size([deg] * order))
    # phi1 = torch.distributions.Uniform(0, 1).sample(torch.Size([b, deg]))
    # phi2 = torch.distributions.Uniform(0, 1).sample(torch.Size([b, deg]))
    # einsum_str = "acde,bd,be->bac"
    # E = torch.einsum(einsum_str, [A, phi1, phi2])
    # print(E.size())
    nl = torch.nn.Sigmoid()
    A = torch.distributions.Uniform(-1, 1).sample(torch.Size([2, 2, 2, 2]))
    out = nl(A)
    err = torch.norm(A-out)
    print(err)
