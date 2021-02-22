# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""

import numpy as np

from .jacobian import jacobian
from .newtonxt import newtonxt


def solve(
    fun,
    x0,
    lpf0,
    jac=None,
    args=(None,),
    dxmax=0.05,
    dlpfmax=0.05,
    control0="lpf",
    jacmode=2,
    jaceps=1e-6,
    maxsteps=80,
    maxcycles=4,
    maxiter=20,
    tol=1e-6,
):
    "Numeric continuation algorithm."

    # init initial control component
    if control0 == "lpf":
        j0 = 1 + len(x0)
    else:
        j0 = control0

    # init y=(x,l)-combined quantities
    y0 = np.append(x0, lpf0)
    dymax = np.append(np.ones_like(x0) * dxmax, dlpfmax)

    # init list for results
    Res = []

    def g(y, n, ymax):
        "Extended equilibrium equations."
        x, l = y[:-1], y[-1]
        return np.append(fun(x, l, *args), np.dot(n, (y - ymax)))

    def dgdy(y, n, ymax):
        "Jacobian of extended equilibrium equations."
        x, l = y[:-1], y[-1]
        if jac is None:
            dfundx = jacobian(fun, argnum=0, mode=jacmode, h=jaceps)
            dfundl = jacobian(fun, argnum=1, mode=jacmode, h=jaceps)
        else:
            dfundx, dfundl = jac
        return np.vstack(
            (np.hstack((dfundx(x, l, *args), dfundl(x, l, *args).reshape(-1, 1))), n)
        )

    # init list for results
    Res = [newtonxt(g, y0, dgdy, j0, dymax, maxiter=0, tol=tol)]

    # Step loop.
    for step in 1 + np.arange(maxsteps):
        ## pre-identification of control component
        res = newtonxt(g, y0, dgdy, j0, dymax, maxiter=1, tol=tol)
        print("\nBegin of Step %d" % step, "\n" + "=" * 68 + "\n")
        print("| Cycle | converged in                        | control component  |")
        print("|" + "-" * 7 + "|" + "-" * 37 + "|" + "-" * 20 + "|")
        # Increment loop.
        for cycl in 1 + np.arange(maxcycles):
            res = newtonxt(g, y0, dgdy, j0, dymax, maxiter=maxiter, tol=tol)
            print(
                "| #{0:4d} | {1:s} | from {2:+4d} to {3:+4d}  |".format(
                    cycl, res.message, j0, res.control
                )
            )

            if res.success:
                if res.control == j0:
                    y0 = res.x
                    Res.append(res)
                    # print(('|'+' '*7+'|'+' '*15+'|'+' '*20+'|\n')*(maxcycles-cycl))
                    print("")
                    print("*final lpf value     = {: .3e}".format(res.x[-1]))
                    print(
                        "*final control value = {: .3e}".format(
                            res.x[abs(res.control) - 1]
                        )
                    )
                    print(
                        "*final equilibrium   = {: .3e} (norm)".format(
                            np.linalg.norm(res.fun)
                        )
                    )
                    break
                else:
                    if cycl == maxcycles:
                        print("\nERROR. Control component changed in last cycle.")
                        print("       Possible solution: Reduce stepwidth.")
                        res.success = False
                    else:
                        # next cycle
                        j0 = res.control
            else:
                break

        if not res.success:
            print("\nERROR. Numerical continuation stopped.")
            break
    return Res
