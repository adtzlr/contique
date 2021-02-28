import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, l, a, k):
    r = a * np.exp(k * l)
    return np.array([-x[0] + r * np.cos(l), -x[1] + r * np.sin(l)])


def test_log_spiral():

    # initial solution
    x0 = np.array([1.0, 0.0])
    lpf0 = 0.0

    # additional function arguments
    a = 1
    k = 0.1

    # numeric continuation
    Res = contique.solve(
        fun=fun,
        x0=x0,
        args=(a, k),
        lpf0=lpf0,
        control0=3,
        dxmax=0.2,
        dlpfmax=0.2,
        jacmode=3,
        jaceps=1e-4,
        maxsteps=500,
        maxcycles=4,
        maxiter=20,
        tol=1e-12,
        overshoot=1.05,
    )

    X = np.array([res.x for res in Res])

    plt.plot(X[:, 0], X[:, 1], "-")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.plot([1], [0], "C0o", lw=3)
    plt.arrow(
        X[-2, 0],
        X[-2, 1],
        X[-1, 0] - X[-2, 0],
        X[-1, 1] - X[-2, 1],
        head_width=1,
        head_length=2,
        fc="C0",
        ec="C0",
    )
    plt.gca().set_aspect("equal")
    plt.savefig("test_log_spiral.svg")


if __name__ == "__main__":
    test_log_spiral()
