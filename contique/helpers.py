# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""

import numpy as np


def needle(j, m):
    "Return zero-vector with needle at component `j` and length `m`."
    n = np.zeros(m)
    n[abs(j) - 1] = 1
    return n


def control(x):
    "Signed index of greatest absolute component of vector x (1-indexed!)."
    j = abs(x).argmax()
    return int((j + 1) * np.sign(x[j]))


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
