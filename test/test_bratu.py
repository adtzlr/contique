import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, lpf):
    n = len(x)
    h = 1 / (n - 1)
    A = np.diag(2 * np.ones_like(x) / h ** 2)
    for i in [1, -1]:
        A -= np.diag(np.ones_like(x[:-1]) / h ** 2, i)
    f = -A.dot(x) + lpf * np.exp(x)
    for i in [0, -1]:
        f[i] = x[i]
    return f


def val(x):
    n = len(x)
    h = 1 / (n - 1)
    H = np.ones(n) * h
    for i in [0, -1]:
        H[i] = h / 2
    return np.sqrt(x.dot(H * x))


def test_bratu():

    # initial solution
    n = 51
    lpf0 = 0.0
    x0 = np.zeros(n)

    # numeric continuation
    # + use rebalance step-width
    # + don't wait until some steps converged until rebalance-increase
    Res = contique.solve(
        fun=fun,
        x0=x0,
        lpf0=lpf0,
        dxmax=0.5,
        dlpfmax=0.5,
        maxsteps=22,
        tol=1e-10,
        rebalance=True,
        minlastfailed=0,
    )

    # extract/calculate load-proportionality factor and unknown u
    lpf = np.array([res.x[-1] for res in Res])
    u = np.array([val(res.x[:-1]) for res in Res])

    # create figure
    plt.plot(lpf, u, "x-")
    plt.xlabel("load-proportionality-factor $LPF$")
    plt.ylabel("$||u||_2$")
    plt.plot(0, 0, "C0o", ms=10)
    plt.savefig("test_bratu.svg")


if __name__ == "__main__":
    test_bratu()
