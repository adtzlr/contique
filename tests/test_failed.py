import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, l, a, b):
    return np.array([-a * np.sin(x[0]) + x[1] ** 2 + l, -b * np.cos(x[1]) * x[1] + l])


def test_sincos_failed():
    # initial solution
    x0 = np.zeros(2)
    lpf0 = 0.0

    # additional function arguments
    a, b = 1, 1

    # numeric continuation
    Res = contique.solve(
        fun=fun,
        x0=x0,
        args=(a, b),
        lpf0=lpf0,
        dxmax=1.0,
        dlpfmax=1.0,
        maxsteps=75,
        maxcycles=4,
        maxiter=3,
        tol=1e-10,
        overshoot=4.0,
        rebalance=True,
        low=0.4,
        solve=np.linalg.solve,
    )

    X = np.array([res.x for res in Res])


if __name__ == "__main__":
    test_sincos_failed()
