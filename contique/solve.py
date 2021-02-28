# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""

import numpy as np

from .jacobian import jacobian
from .newtonxt import newtonxt
from . import printinfo


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
    overshoot=1.0,
):
    "Numeric continuation algorithm."

    # init number of unknows
    ncomp = 1 + len(x0)

    # initial control component
    if control0 == "lpf":
        j0 = ncomp
    else:
        j0 = control0

    # init y=(x,l)-combined quantities
    y0 = np.append(x0, lpf0)
    dymax = np.append(np.ones_like(x0) * dxmax, dlpfmax)

    # init list of results
    Res = [newtonxt(fun, jac, y0, j0, dymax, jacmode, jaceps, args, maxiter=0, tol=tol)]

    printinfo.header()

    # Step loop.
    for step in 1 + np.arange(maxsteps):
        ## pre-identification of control component
        res = newtonxt(
            fun, jac, y0, j0, dymax, jacmode, jaceps, args, maxiter=1, tol=tol
        )

        # Cycle loop.
        for cycl in 1 + np.arange(maxcycles):

            # Newton Iterations.
            res = newtonxt(
                fun, jac, y0, j0, dymax, jacmode, jaceps, args, maxiter=maxiter, tol=tol
            )
            printinfo.cycle(
                step,
                cycl,
                j0,
                res.control,
                res.status,
                np.linalg.norm(res.fun),
                res.niterations,
                max(abs(res.dys)) <= overshoot,
            )

            # Did Newton Iterations converge?
            if res.success:

                # Did control component change? OR Was overshoot inside allowed range?
                if (res.control == j0) or (max(abs(res.dys)) <= overshoot):

                    # Save results, move to next step.
                    j0 = res.control
                    y0 = res.x
                    Res.append(res)
                    break

                else:  # Were max. number of cycles reached?
                    if cycl == maxcycles:
                        # Print Error and set success-flag to False.
                        printinfo.errorcontrol()
                        res.success = False
                    else:
                        # re-cycle Step with new control component
                        j0 = res.control
            else:
                # break cycle loop if Newton Iterations failed.
                break

        # break step loop if Newton Iterations failed.
        if not res.success:
            printinfo.errorfinal()
            break

    return Res
