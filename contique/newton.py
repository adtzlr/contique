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

from .helpers import argparser


class NewtonResult:
    """Class for handling the results of a Newton-Rhapson solution.

    This class has several public attribues.

    Attributes
    ----------
    success : bool
        flag for the converged solution
    message : str
        message for the state
    status : int
        integer representig the status of the solution (0 if not converged, 1 if converged).
    niterations : int
        number of performed iterations
    x : ndarray
        1d-array containing the unknows
    fun : function
        function returning the equilibrium equations
    jac : function
        function returning the jacobian of the equilibrium equations

    """

    def __init__(self, fun, x0, jac, args):
        """Initialize an Instance of a NewtonResult.

        Parameters
        ----------
        fun : function
            function returning the equilibrium equations
        x0 : ndarray
            1d-array containing the initial unknows
        jac : function
            function returning the jacobian of the equilibrium equations
        args : tuple, optional
            Optional tuple of arguments which are passed to the function. Eeven if only
            one argument is passed, it has to be encapsulated in a tuple (default is (None,)).

        """
        self.success = False
        self.message = "not started"
        self.status = 0
        self.niterations = 0
        self.x = x0.copy()
        self.fun = argparser(fun)(self.x, *args)
        self.jac = argparser(jac)(self.x, *args)


def newtonrhapson(fun, x0, jac, args=(None,), maxiter=8, tol=1e-8):
    """A simple n-dimensional Newton-Rhapson solver.

    Parameters
    ----------
    fun : function
        function in terms of unknows x and optional args which returns the
        equilibrium equations.
    x0 : ndarray
        1d-array with initial values of unknows x
    jac : function
        jacobian of fun w.r.t. the unknows x
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Eeven if only
        one argument is passed, it has to be encapsulated in a tuple (default is (None,)).
    maxiter : int, optional
        maximum number of iterations (default is 8)
    tol : float, optional
        tolerated residual of the norm of the equilibrium equation (default is 1e-8)

    Returns
    -------
    res : NewtonResult
        Instance of NewtonResult with res.x being the final unknowns

    """

    # init result object with initial function evaluation
    res = NewtonResult(fun, x0, jac, args)

    # iteration loop
    for res.niterations in range(1, 1 + maxiter):

        # solve linear equation system
        res.x += np.linalg.solve(res.jac, -res.fun)

        # calculate function and jacobian at updated x
        res.fun = argparser(fun)(res.x, *args)
        res.jac = argparser(jac)(res.x, *args)

        # convergence check
        if np.linalg.norm(res.fun) < tol:
            res.success = True
            res.status = 1
            res.message = "Solution converged in {0:2d} Iteration".format(
                res.niterations
            )
            if res.niterations > 1:
                res.message = res.message + "s"

            break

    # check if newton process failed
    if not res.success:
        if maxiter == 1:
            res.message = "Calculated linear solution because of input parameter `maxiter=1` (not converged)."
        else:
            res.message = "Newton-R. process failed."

    return res
