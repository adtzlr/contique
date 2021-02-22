# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""
from types import SimpleNamespace
import copy
import numpy as np


def jacobian(fun, argnum=0, h=1e-6, mode=2):
    """Decorator for the jacobian as 2- or 3-point finite-differences approximation
    w.r.t. a given argnum and h.

    Parameters
    ----------
    fun : function
        Function for which the jacobian should be approximated.
    argnum : int
        Evaluate the jacobian w.r.t the the selected argument (default is 0).
    h : float
        A small number (default is 1e-6).
    mode : int

    Returns
    -------
    jacwrapper : function
        Function for the calculation of the jacobian of function `fun` w.r.t. given `argnum`.
    """

    def jacwrapper(*args, **kwargs):
        """Calculates the jacobian as 2- or 3-point finite-differences approximation
        w.r.t. a given argnum and h."""

        # pre-evaluate f0 = f(x0) if 2-point scheme is used
        if mode == 2:
            f0 = fun(*args, **kwargs)

        # check if arg is an array
        if isinstance(args[argnum], np.ndarray):
            # init jacobian
            nargs = np.size(args[argnum])
            jac = np.zeros((nargs, nargs))

            # loop over columns
            for j in range(nargs):

                # copy args and modify item j of 1d-args
                fwdargs = copy.deepcopy(args)
                fwdargs[argnum].ravel()[j] = fwdargs[argnum].ravel()[j] + h
                if mode == 3:
                    rvsargs = copy.deepcopy(args)
                    rvsargs[argnum].ravel()[j] = rvsargs[argnum].ravel()[j] - h
                    f0 = fun(*rvsargs, **kwargs)
                jac[:, j] = (fun(*fwdargs, **kwargs) - f0) / h / (mode - 1)
            jac = jac.reshape(*args[argnum].shape, *args[argnum].shape)

        else:  # arg is float
            # allow item assignment (convert tuple of args to list)
            fwdargs = list(copy.deepcopy(args))
            fwdargs[argnum] = fwdargs[argnum] + h
            if mode == 3:
                rvsargs = list(copy.deepcopy(args))
                rvsargs[argnum] = rvsargs[argnum] - h
                f0 = fun(*rvsargs, **kwargs)
            # calculate jacobian
            jac = (fun(*fwdargs, **kwargs) - f0) / h / (pts - 1)
        return jac

    return jacwrapper


def needle(j, m):
    "Vector with needle at component j and dimension m."
    n = np.zeros(m)
    n[abs(j) - 1] = 1
    return n


def control(x):
    "Signed index of greatest absolute component of vector x (1-indexed!)."
    j = abs(x).argmax()
    return int((j + 1) * np.sign(x[j]))


def newtonrhapson(fun, x0, jac, args=(None,), maxiter=8, tol=1e-8):
    "Default Newton-Rhapson solver."
    res = SimpleNamespace()
    res.success, res.message, res.x = False, "not started", x0.copy()
    res.fun = fun(res.x, *args)

    iteration = 0
    for iteration in range(maxiter):
        res.jac = jac(res.x, *args)
        res.x += np.linalg.solve(res.jac, -res.fun)
        res.fun = fun(res.x, *args)

        if np.linalg.norm(res.fun) < tol:
            res.success = True
            if iteration == 1:
                res.message = "{0:2d} Iteration ".format(iteration)
            else:
                res.message = "{0:2d} Iterations".format(iteration)
            break

    if not res.success and maxiter > 1:
        res.message = "Newton failed"

    res.niterations = iteration + 1
    return res


def newtonxt(g, y0, dgdy, control0, dymax, maxiter=20, tol=1e-8):
    """Solve equilibrium equations starting from an initial solution
    with control component and max. allowed increase of unknowns."""
    n = needle(control0, len(y0))
    ymax = y0 + np.sign(control0) * dymax
    res = newtonrhapson(
        fun=g, x0=y0, jac=dgdy, args=(n, ymax), maxiter=maxiter, tol=tol
    )
    res.dys = (res.x - y0) / dymax
    res.control = control(res.dys)
    return res


def stepcontrol(increase=2, decrease=0.1):
    # last 3: good progress? increase
    return


def numcont(
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
            dfundx = jacobian(fun, argnum=0, pts=jacmode, h=jaceps)
            dfundl = jacobian(fun, argnum=1, pts=jacmode, h=jaceps)
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
        print("\nBegin of Step %d" % step, "\n" + "=" * 46 + "\n")
        print("| Cycle | converged in  | control component  |")
        print("|" + "-" * 7 + "|" + "-" * 15 + "|" + "-" * 20 + "|")
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


if __name__ == "__main__":

    # def f(x,l):
    # return np.array([-x[0]-np.sin(x[0])*(1+x[0]*1)+l])

    def f(x, l, a, b):
        return np.array(
            [-a * np.sin(x[0]) + x[1] ** 2 + l, -b * np.cos(x[1]) * x[1] + l]
        )

    # initial solution
    x0 = np.zeros(2)
    lpf0 = 0.0

    # additional function arguments
    a, b = 1, 1

    # numeric continuation
    Res = numcont(
        fun=f,
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

    # Res = numcont(fun=f,x0=x0,lpf0=lpf0,jac=None,jacmode=2,
    #               dxmax=0.05,dlpfmax=0.05,control0='lpf',
    #               maxsteps=80,maxcycles=4,maxiter=20,tol=1e-6)

    X = np.array([res.x for res in Res])

    import matplotlib.pyplot as plt

    plt.plot(X[:, 0], X[:, 1], ".-")
    plt.show()
