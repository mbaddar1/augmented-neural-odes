import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


class vdp_func2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, x):
        mio = 0.5
        a = 1.2
        w = 2 * 3.14 / 10

        x1 = x[0]
        x2 = x[1]
        x1_dot = x2
        x2_dot = mio * (1 - x1 ** 2) * x2 - x1 + \
                 a * np.sin(w * t)
        y = np.zeros(2)
        y[0] = x1_dot
        y[1] = x2_dot
        return y


if __name__ == '__main__':
    s = solve_ivp(fun=vdp_func2(), t_span=(300, 600), y0=[0, 0],t_eval=np.arange(300,600,0.1))
    print(s.y)
    plt.plot(s.t, s.y[0])
    plt.savefig('vdp_y1_t.png')
    plt.clf()
    #
    plt.plot(s.t, s.y[1])
    plt.savefig('vdp_y2_t.png')
    plt.clf()
    #
    plt.plot(s.y[0], s.y[1])
    plt.savefig('vdp_ph.png')
