import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, l, a):
    r = a / np.sqrt(l)
    return np.array([-x[0] + r * np.cos(l), -x[1] + r * np.sin(l)])


def test_lituus_spiral():

    # additional function arguments
    a = 1

    # initial solution
    lpf0 = 0.2
    x0 = fun(np.zeros(2), lpf0, a)

    # lpf0 = 1.0
    # x0 = np.array([1.,0.])

    # numeric continuation
    Res = contique.solve(
        fun=fun,
        x0=x0,
        args=(a,),
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
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.plot(X[0, 0], X[0, 1], "C0o", lw=3)
    # plt.arrow(X[-2,0],X[-2,1],X[-1,0]-X[-2,0],X[-1,1]-X[-2,1],
    #          head_width=0.1, head_length=0.2, fc='C0', ec='C0')
    plt.gca().set_aspect("equal")
    plt.savefig("test_lituus_spiral.svg")


if __name__ == "__main__":
    test_lituus_spiral()
