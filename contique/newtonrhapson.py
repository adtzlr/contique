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
    def __init__(self, fun, x0, jac, args):
        self.success = False
        self.message = "not started"
        self.x = x0.copy()
        self.fun = argparser(fun)(self.x, *args)
        self.jac = argparser(jac)(self.x, *args)
        self.iteration = 0
        self.niterations = 0
        self.status = 0


def newtonrhapson(fun, x0, jac, args=(None,), maxiter=8, tol=1e-8):
    "A simple n-dimensional Newton-Rhapson solver."

    # init result object with initial function evaluation
    res = NewtonResult(fun, x0, jac, args)

    # iteration loop
    for res.iteration in range(maxiter):

        # solve linear equation system
        res.x += np.linalg.solve(res.jac, -res.fun)

        # calculate function and jacobian at updated x
        res.fun = argparser(fun)(res.x, *args)
        res.jac = argparser(jac)(res.x, *args)

        # convergence check
        if np.linalg.norm(res.fun) < tol:
            res.success = True
            res.status = 1
            res.message = "Solution converged in {0:2d} Iteration".format(res.iteration)
            if res.iteration > 1:
                res.message = res.message + "s"

            break

    # check if newton process failed
    if not res.success:
        if maxiter == 1:
            res.message = "Calculated linear solution because of input parameter `maxiter=1` (not converged)."
        else:
            res.message = "Newton-R. process failed."

    # set number of performed iterations
    res.niterations = res.iteration + 1

    return res
