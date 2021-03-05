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
import copy
import numpy as np


def jacobian(fun, argnum=0, h=None, mode=3):
    """Decorator for the jacobian as 2- or 3-point finite-differences
    approximation w.r.t. a given argnum and h.

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
        Function for the calculation of the jacobian of function `fun`
        w.r.t. given `argnum`.
    """

    # set optimal step-width
    if h is None:
        h = ((np.finfo(float).eps)) ** (1 / mode)

    def jacwrapper(*args, **kwargs):
        """Calculates the jacobian as 2- or 3-point finite-differences
        approximation w.r.t. a given argnum and h."""

        # pre-evaluate f0 = f(x0) if 2-point scheme is used
        f0 = fun(*args, **kwargs)

        # check if arg is an array
        if isinstance(args[argnum], np.ndarray):
            # init 2d-jacobian
            nargs = np.size(args[argnum])
            nfuns = np.size(f0)
            jac = np.zeros((nfuns, nargs))

            # loop over columns
            for j in range(nargs):

                # copy args and modify item j of 1d-args
                fwdargs = copy.deepcopy(args)
                fwdargs[argnum].ravel()[j] = fwdargs[argnum].ravel()[j] + h

                f = fun(*fwdargs, **kwargs)

                # re-define f0
                if mode == 3:
                    rvsargs = copy.deepcopy(args)
                    rvsargs[argnum].ravel()[j] = rvsargs[argnum].ravel()[j] - h
                    f0 = fun(*rvsargs, **kwargs)

                jac[:, j] = (f - f0).ravel() / h / (mode - 1)

            # reshape 2d-jacobian to desired shape
            jac = jac.reshape(*f0.shape, *args[argnum].shape)

        else:  # arg is float
            # allow item assignment (convert tuple of args to list)
            fwdargs = list(copy.deepcopy(args))
            fwdargs[argnum] = fwdargs[argnum] + h

            f = fun(*fwdargs, **kwargs)

            # re-define f0
            if mode == 3:
                rvsargs = list(copy.deepcopy(args))
                rvsargs[argnum] = rvsargs[argnum] - h
                f0 = fun(*rvsargs, **kwargs)

            # calculate jacobian
            jac = (f - f0) / h / (mode - 1)

        return jac

    return jacwrapper
