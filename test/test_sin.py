import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, l, a, b):
    return np.array([-(a+b*x[0]) * np.sin(x[0]) + l])


def test_sin():

    # initial solution
    x0 = np.zeros(1)
    lpf0 = 0.0

    # additional function arguments
    a = 1
    b = 0.3

    # numeric continuation
    Res = contique.solve(
        fun=fun,
        x0=x0,
        args=(a, b),
        lpf0=lpf0,
        dxmax=0.1,
        dlpfmax=0.1,
        maxsteps=500,
        maxcycles=4,
        maxiter=20,
        tol=1e-10,
        overshoot=1.05,
        rebalance=True,
        high=10
    )

    X = np.array([res.x for res in Res])

    plt.plot(X[:, 0], X[:, 1], "C0.-")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.plot([0], [0], "C0o", lw=3)
    plt.arrow(
        X[-2, 0],
        X[-2, 1],
        X[-1, 0] - X[-2, 0],
        X[-1, 1] - X[-2, 1],
        head_width=0.075,
        head_length=0.15,
        fc="C0",
        ec="C0",
    )
    plt.gca().set_aspect("equal")
    plt.savefig("test_sin.svg")


if __name__ == "__main__":
    test_sin()
