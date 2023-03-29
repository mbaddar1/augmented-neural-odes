import torch

if __name__ == '__main__':
    S = torch.tensor([[1.0, 2], [-1, 4]])
    Lambda = torch.diag(input=torch.tensor([-0.1, -0.2]))
    A = torch.matmul(S, Lambda)
    A = torch.matmul(A, torch.inverse(S))
    e1 = torch.linalg.eigvals(A)
    print(e1)