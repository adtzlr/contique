# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: adtzlr

Contique - Numeric continuation of equilibrium equations
Copyright (C) 2022 Andreas Dutzler

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
from scipy import sparse

from .jacobian import jacobian
from .helpers import needle, control
from .newton import newtonrhapson


def funxt(y, needle_vector, ymax, fun, jac=None, jacmode=3, jaceps=None, args=(None,)):
    """Extend the given equilibrium equations.

    Parameters
    ----------
    y : array
        1d-array of unknowns
    needle_vector : array
        (pre-evaluated) needle-vector
    ymax : array
        1d-array with max. allowed values of unknows
    fun : function
        1d-array of equilibrium equations
    jac : function, optional
        jacobian of fun w.r.t. the extended unknows y
    jacmode : int, optional
        forward (2) or central (3) finite-differences approx. of the jacobian
    jaceps : float, optional
        user-specified stepwidth (if None, this defaults to eps^(1/mode))
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Even if
        only one argument is passed, it has to be encapsulated in a tuple
        (default is (None,)).

    Returns
    -------
    array
        extended 1d-array of equilibrium equations
        with control equation
    """

    # split the unknowns
    x, l = y[:-1], y[-1]

    # evaluate the given function
    f = fun(x, l, *args)

    if sparse.issparse(f):
        # convert function vector to array
        f = f.toarray()

    # extend the function
    return np.append(f, np.dot(needle_vector, (y - ymax)))


def jacxt(y, needle_vector, ymax, fun, jac=None, jacmode=3, jaceps=None, args=(None,)):
    """Jacobian of extended equilibrium equations.

    Parameters
    ----------
    y : ndarray
        1d-array of extended unknows
    needle_vector : ndarray
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
        Optional tuple of arguments which are passed to the function. Even if
        only one argument is passed, it has to be encapsulated in a tuple
        (default is (None,)).

    Returns
    -------
        ndarray
        jacobian of fun w.r.t. y (contains both derivatives of x and lpf)
        as 2d-array
    """

    # split the unknowns
    x, lpf = y[:-1], y[-1]

    if jac is None:
        # evaluate by finite differences method
        dfundx = jacobian(fun, argnum=0, mode=jacmode, h=jaceps)
        dfundl = jacobian(fun, argnum=1, mode=jacmode, h=jaceps)
    else:
        dfundx, dfundl = jac

    # evaluate the given jacobian
    dfdx = dfundx(x, lpf, *args)
    dfdl = dfundl(x, lpf, *args).reshape(-1, 1)

    # define horizontal and vertical stack operations based on evaluated
    # sparse or dense jacobian
    if sparse.issparse(dfdx):
        hstack = sparse.hstack
        vstack = sparse.vstack
        array = sparse.csr_matrix
    else:
        hstack = np.hstack
        vstack = np.vstack
        array = np.array

    # extend the jacobian
    dfdy = hstack([array(dfdx), array(dfdl)])
    dgdy = vstack([dfdy, array(needle_vector)])

    if sparse.issparse(dfdx):
        # convert to compressed sparse row format
        dgdy = dgdy.tocsr()

    return dgdy


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
    solve=None,
):
    """Solve equilibrium equations starting from an initial solution
    with a given control component and a max. allowed increase of unknowns.

    Parameters
    ----------
    fun : function
        function in terms of extended unknows and optional args which returns
        the extended equilibrium equations
    jac : function, optional
        jacobian of fun w.r.t. the extended unknows
    y0 : ndarray
        1d-array of initial extended unknows
    control0 : tuple of int, optional
        initial tuple of control component and sign
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
    solve: callable, optional
        A solver.

    Returns
    -------
    res : NewtonResult
        Instance of NewtonResult with res.x being the final extended unknowns
    """

    # init needle-vector and obtain ymax
    component0, sign0 = control0
    n = needle(component0, len(y0))
    ymax = y0 + sign0 * dymax

    # Newton-Rhapson solver
    res = newtonrhapson(
        fun=funxt,
        x0=y0,
        jac=jacxt,
        args=(n, ymax, fun, jac, jacmode, jaceps, args),
        maxiter=maxiter,
        tol=tol,
        solve=solve,
    )

    # normalized dy = dy/dymax
    res.dys = (res.x - y0) / dymax

    # final control component based on normalized dy
    if np.any(np.isnan(res.dys)):
        res.control = control0
    else:
        res.control = control(res.dys)

    return res
