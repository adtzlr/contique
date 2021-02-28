# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""
import numpy as np

from .jacobian import jacobian
from .helpers import needle, control
from .newtonrhapson import newtonrhapson


def funxt(y, n, ymax, fun, jac, jacmode, jaceps, args):
    """Extended equilibrium equations.

    Parameters
    ----------
    y : array
        1d-array of unknowns.
    n : array
        pre-evaluated needle-vector.
    ymax : array
        1d-array with max. allowed values of unknows.
    fun : function
        1d-array of equilibrium equations
    jac : function
        jacobian of equilibrium euqations (not used)
    jacmode : int
        2 or 3-point evaulation of the jacobian (not used)
    jaceps : float
        a small number to evaulate the jacobian (not used)
    args : tuple
        optional function arguments

    Returns
    -------
    array
        extended 1d-array of equilibrium equations
        with control equation
    """

    x, l = y[:-1], y[-1]
    return np.append(fun(x, l, *args), np.dot(n, (y - ymax)))


def jacxt(y, n, ymax, fun, jac, jacmode, jaceps, args):
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


def newtonxt(
    fun,
    jac,
    y0,
    control0,
    dymax,
    jacmode=3,
    jaceps=1e-6,
    args=(None,),
    maxiter=20,
    tol=1e-8,
):
    """Solve equilibrium equations starting from an initial solution
    with control component and max. allowed increase of unknowns."""

    n = needle(control0, len(y0))
    ymax = y0 + np.sign(control0) * dymax
    res = newtonrhapson(
        fun=funxt,
        x0=y0,
        jac=jacxt,
        args=(n, ymax, fun, jac, jacmode, jaceps, args),
        maxiter=maxiter,
        tol=tol,
    )
    res.dys = (res.x - y0) / dymax
    res.control = control(res.dys)
    return res
