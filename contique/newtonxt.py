# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""
import numpy as np

from .helpers import needle, control
from .newtonrhapson import newtonrhapson


def newtonxt(g, y0, dgdy, control0, dymax, maxiter=20, tol=1e-8):
    """Solve equilibrium equations starting from an initial solution
    with control component and max. allowed increase of unknowns."""

    n = needle(control0, len(y0))
    ymax = y0 + np.sign(control0) * dymax
    res = newtonrhapson(
        fun=g, x0=y0, jac=dgdy, args=(n, ymax), maxiter=maxiter, tol=tol
    )
    res.dys = (res.x - y0) / dymax
    res.control = control(res.dys)
    return res
