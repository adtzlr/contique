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
):
    "Numeric continuation algorithm."

    # init initial control component
    ncomp = 1 + len(x0)
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

        # Increment loop.
        for cycl in 1 + np.arange(maxcycles):
            res = newtonxt(
                fun, jac, y0, j0, dymax, jacmode, jaceps, args, maxiter=maxiter, tol=tol
            )
            printinfo.cycle(
                step, cycl, j0, res.control, res.status, np.linalg.norm(res.fun)
            )

            if res.success:
                if res.control == j0:
                    y0 = res.x
                    Res.append(res)
                    break
                else:
                    if cycl == maxcycles:
                        printinfo.errorcontrol()
                        res.success = False
                    else:
                        # next cycle
                        j0 = res.control
            else:
                break

        if not res.success:
            printinfo.errorfinal()
            break
    return Res
