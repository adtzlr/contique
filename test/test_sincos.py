import matplotlib.pyplot as plt
import pytest

import contigue


def fun(x, l, a, b):
    return np.array([-a * np.sin(x[0]) + x[1] ** 2 + l, -b * np.cos(x[1]) * x[1] + l])


def test_sincos():

    # initial solution
    x0 = np.zeros(2)
    lpf0 = 0.0

    # additional function arguments
    a, b = 1, 1

    # numeric continuation
    Res = contigue.solve(
        fun=fun,
        x0=x0,
        args=(a, b),
        lpf0=lpf0,
        dxmax=0.05,
        dlpfmax=0.05,
        maxsteps=80,
        maxcycles=4,
        maxiter=20,
        tol=1e-6,
    )

    X = np.array([res.x for res in Res])

    plt.plot(X[:, 0], X[:, 1], ".-")
