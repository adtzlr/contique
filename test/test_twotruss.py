import matplotlib.pyplot as plt
import numpy as np
import pytest

import contique


def fun(x, lpf, a, L, EA):
    WL = -x[0] / L
    lL = np.sqrt(1 - 2 * np.sin(a) * WL + WL ** 2)
    N = EA * (lL - 1)
    return np.array([2 * N * (np.sin(a) - WL) + lpf])


def test_twotruss():

    # initial solution
    x0 = np.zeros(1)
    lpf0 = 0.0

    # args
    L = np.sqrt(2)
    a = np.deg2rad(45)
    EA = 1

    # numeric continuation
    Res = contique.solve(fun=fun, x0=x0, lpf0=lpf0, args=(a, L, EA))

    X = np.array([res.x for res in Res])
    print(X[-2:, :])

    plt.plot(X[:, 0], X[:, 1], ".-")
    plt.xlabel("$W$")
    plt.ylabel("$LPF$")
    plt.plot([0], [0], "C0o", lw=3)
    plt.arrow(
        X[-2, 0],
        X[-2, 1],
        X[-1, 0] - X[-2, 0],
        X[-1, 1] - X[-2, 1],
        head_width=0.05,
        head_length=0.1,
        fc="C0",
        ec="C0",
    )
    plt.gca().set_aspect("equal")
    plt.savefig("test_twotruss.svg")


if __name__ == "__main__":
    test_twotruss()
