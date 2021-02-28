# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""
import numpy as np


def argparser(fun):
    def inner(x, *args, **kwargs):
        if args[0] is None and kwargs is None:
            f = fun(x)
        if args[0] is not None and kwargs is None:
            f = fun(x, *args)
        if args[0] is None and kwargs is not None:
            f = fun(x, **kwargs)
        else:  # args[0] is not None and kwargs is not None:
            f = fun(x, *args, **kwargs)
        return f

    return inner


class ResultObject:
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
    res = ResultObject(fun, x0, jac, args)

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
            if res.iteration == 1:
                res.message = "Solution converged in {0:2d} Iteration ".format(
                    res.iteration
                )
                res.status = 1
            else:
                res.message = "Solution converged in {0:2d} Iterations".format(
                    res.iteration
                )
                res.status = 1
            break

    # check if newton process failed
    if not res.success:
        if maxiter == 1:
            res.message = "Calculated linear solution because of input parameter `maxiter=1` (not converged)."
        else:
            res.message = "Newton-R. process failed."

    # set number of iterations
    res.niterations = res.iteration + 1

    return res
