# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: adtzlr

Contique - Numeric continuation of equilibrium equations
Copyright (C) 2021 Andreas Dutzler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

from .helpers import argparser2
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
    jacmode=3,
    jaceps=None,
    maxsteps=50,
    maxcycles=4,
    maxiter=8,
    tol=1e-6,
    overshoot=1.0,
    rebalance=False,
    increase=0.5,
    decrease=2.0,
    high=10,
    low=1e-6,
    minlastfailed=3,
):
    """Numeric continuation of (nonlinear) equilibrium equations.

    Parameters
    ----------
    fun : function
        function in terms of unknows x and optional args which returns the
        equilibrium equations.
    x0 : ndarray
        1d-array with initial values of unknows x
    lpf0 : float
        initial value for the load-proportionality-factor
    jac : function, optional
        jacobian of fun w.r.t. the unknows x
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Eeven if only
        one argument is passed, it has to be encapsulated in a tuple (default is (None,)).
    dxmax : float, optional
        max. allowed absolute incremental increase of unknowns per step
    dlpfmax : float, optional
        max. allowed absolute incremental increase of lpf per step
    control0 : int, optional
        initial signed control component ( 1-indexed )
    jacmode : int, optional
        forward (2) or central (3) finite-differences approx. of the jacobian
    jaceps : float, optional
        user-specified stepwidth (if None, this defaults to eps^(1/mode))
    maxsteps : int, optional
        max. number of steps
    maxcycles : int, optional
        max. number of cycles per step
    maxiter : int, optional
        max. number of Newton-iterations per cycle
    tol : float, optional
        tolerated residual of the norm of the equilibrium equation (default is 1e-8)
    overshoot : float, optional
        allowed overshoot of the final control component of a cycle (default is 1.0)
    rebalance : bool, optional
        rebalance max. allowed incremental increase values after each step
    increase : float, optional
        rebalance increase factor
    decrease : float, optional
        rebalance decrease factor
    high : float, optional
        rebalance max. factor of incremental increase w.r.t. to the initial values
    low : float, optional
        rebalance min. factor of incremental increase w.r.t. to the initial values
    minlastfailed : int, optional
        rebalance increase only after a given number of converged steps

    Returns
    -------
    Res : list
        List of NewtonResults (with res.x being the final unknowns per step)

    """

    # allow passing empty *args to fun(x, lpf)
    fun = argparser2(fun)

    # init number of unknows
    ncomp = 1 + len(x0)

    # init rebalance and lastfailed
    # (not used if not rebalance)
    rebalanced = False
    lastfailed = 0

    # initial control component
    if control0 == "lpf":
        j0 = ncomp
    else:
        j0 = control0

    # init y=(x,l)-combined quantities
    y0 = np.append(x0, lpf0)
    dymax = np.append(np.ones_like(x0) * dxmax, dlpfmax)
    dymax0 = dymax.copy()

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

        # Rebalance max. incremental unknowns
        # --------------------------------------------------------------------
        if rebalance:
            dymaxn = dymax.copy()
            dymax, rebalanced, lastfailed = adjust(
                dymax0,
                dymaxn,
                success=res.success,
                n=res.niterations,
                lastfailed=lastfailed,
                increase=increase,
                decrease=decrease,
                high=high,
                low=low,
                minlastfailed=minlastfailed,
                nref=8,
            )
        # --------------------------------------------------------------------

        # break step loop if Newton Iterations failed.
        if not res.success and not rebalanced:
            printinfo.errorfinal()
            break

    return Res


def adjust(
    x0,
    xn,
    success,
    n,
    lastfailed,
    increase=0.5,
    decrease=4.0,
    high=10,
    low=1e-6,
    minlastfailed=3,
    nref=8,
):
    "Adjust (rebalance) the stepwidth. **Warning**: Experimental feature."

    rebalanced = True
    if success:
        lastfailed += 1
        if lastfailed >= minlastfailed:
            x = xn * (1 + (nref - min(n, nref)) / nref * increase)
        else:  # lastfailed < minlastfailed
            x = xn

    elif not success:
        x = xn / decrease
        lastfailed = 0

    y = np.maximum(np.minimum(x / x0, high), low) * x0
    if y[0] == xn[0]:
        rebalanced = False

    return y, rebalanced, lastfailed
