import torch

if __name__ == '__main__':
    X = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    A = torch.Tensor([[0.9, 0.8], [0.6, -0.4]])
    Y = torch.einsum('ji,bi->bj',A,X)
    A1 = torch.linalg.lstsq(X,Y).solution.T
    print(A1)