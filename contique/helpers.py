# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:31:04 2021

@author: Andreas
"""

import numpy as np


def needle(j, m):
    "Vector with needle at component j and dimension m."
    n = np.zeros(m)
    n[abs(j) - 1] = 1
    return n


def control(x):
    "Signed index of greatest absolute component of vector x (1-indexed!)."
    j = abs(x).argmax()
    return int((j + 1) * np.sign(x[j]))
