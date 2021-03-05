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

from .jacobian import jacobian
from .helpers import needle, control
from .newton import newtonrhapson


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


def jacxt(y, n, ymax, fun, jac=None, jacmode=3, jaceps=None, args=(None,)):
    """Jacobian of extended equilibrium equations.

    Parameters
    ----------
    y : ndarray
        1d-array of extended unknows
    n : ndarray
        1d-array with pre-evaluated needle-vector
    ymax : ndarray
        1d-array with max. allowed incremental increase values of y
    fun : function
        function in terms of unknows x and optional args which returns the
        equilibrium equations.
    jac : function, optional
        jacobian of fun w.r.t. the extended unknows y
    jacmode : int, optional
        forward (2) or central (3) finite-differences approx. of the jacobian
    jaceps : float, optional
        user-specified stepwidth (if None, this defaults to eps^(1/mode))
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Eeven if only
        one argument is passed, it has to be encapsulated in a tuple (default is (None,)).

    Returns
    -------
        ndarray
        jacobian of fun w.r.t. y (contains both derivatives of x and lpf) as 2d-array
    """
    x, lpf = y[:-1], y[-1]
    if jac is None:
        dfundx = jacobian(fun, argnum=0, mode=jacmode, h=jaceps)
        dfundl = jacobian(fun, argnum=1, mode=jacmode, h=jaceps)
    else:
        dfundx, dfundl = jac
    return np.vstack(
        (np.hstack((dfundx(x, lpf, *args), dfundl(x, lpf, *args).reshape(-1, 1))), n)
    )


def newtonxt(
    fun,
    jac,
    y0,
    control0,
    dymax,
    jacmode=3,
    jaceps=None,
    args=(None,),
    maxiter=20,
    tol=1e-8,
):
    """Solve equilibrium equations starting from an initial solution
    with control component and max. allowed increase of unknowns.

    Parameters
    ----------
    fun : function
        function in terms of unknows x and optional args which returns the
        equilibrium equations.
    jac : function, optional
        jacobian of fun w.r.t. the extended unknows y
    y0 : ndarray
        1d-array of initial extended unknows
    control0 : int, optional
        initial signed control component ( 1-indexed )
    dxmax : float, optional
        max. allowed absolute incremental increase of extended unknowns per step
    jacmode : int, optional
        forward (2) or central (3) finite-differences approx. of the jacobian
    jaceps : float, optional
        user-specified stepwidth (if None, this defaults to eps^(1/mode))
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Eeven if only
        one argument is passed, it has to be encapsulated in a tuple (default is (None,)).
    maxiter : int, optional
        max. number of Newton-iterations per cycle
    tol : float, optional
        tolerated residual of the norm of the equilibrium equation (default is 1e-8)

    Returns
    -------
    res : NewtonResult
        Instance of NewtonResult with res.x being the final extended unknowns
    """

    # init needle-vector and obtain ymax
    n = needle(control0, len(y0))
    ymax = y0 + np.sign(control0) * dymax

    # Newton-Rhapson solver
    res = newtonrhapson(
        fun=funxt,
        x0=y0,
        jac=jacxt,
        args=(n, ymax, fun, jac, jacmode, jaceps, args),
        maxiter=maxiter,
        tol=tol,
    )

    # normalized dy = dy/dymax
    res.dys = (res.x - y0) / dymax

    # final control component based on dnormalized dy
    res.control = control(res.dys)

    return res
