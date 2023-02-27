"""
This py script is for experimentation and validation if my understanding for Matrix ODE
"""
import numpy
import numpy as np
import scipy as scipy
import torch

from phd_experiments.torch_ode_solvers.torch_euler import TorchEulerSolver
from phd_experiments.torch_ode_solvers.torch_rk45 import TorchRK45

"""
Refs
https://www.simiode.org/resources/6425/download/5-010-Text-S-MatrixExponential-StudentVersion.pdf 
"""
from scipy.integrate import solve_ivp


def ode_func(t, x, A):
    return numpy.matmul(A, x)


if __name__ == '__main__':
    z0 = np.array([0.1, 0.2])
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    T = 1.0
    tensor_dtype = torch.float64
    # use scipy
    soln = solve_ivp(fun=ode_func, t_span=(0, T), y0=z0, args=(A,))
    zT_scipy = soln.y[:, -1]

    # use expm
    E = scipy.linalg.expm(A * T)
    zT_expm = np.matmul(E, z0)

    # use my torch euler
    torch_euler_solve = TorchEulerSolver(step_size=0.1)
    soln = torch_euler_solve.solve_ivp(func=ode_func, t_span=(0, T), z0=torch.Tensor(z0), args=(A,))
    zT_torch_euler = soln.z_trajectory[-1].detach().numpy()

    # use my rk45 torch solver

    torch_rk45_solver = TorchRK45(device=torch.device('cpu'), tensor_dtype=torch.float64)
    soln = torch_rk45_solver.solve_ivp(func=ode_func, t_span=(0, T), z0=torch.tensor(z0, dtype=tensor_dtype),
                                       args=(torch.tensor(A, dtype=tensor_dtype),))
    zt_torch_rk45 = soln.z_trajectory[-1]
    print("")
