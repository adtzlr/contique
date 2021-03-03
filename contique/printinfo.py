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


def header():
    print("|Step,C.| Control Comp. | Norm (Iter.#) | Message     |")
    print("|-------|---------------|---------------|-------------|")


def cycle(step, cycl, j0, control, status, fnorm, niterations, overshootcond):
    if cycl > 1:
        stp = "     "
    else:
        stp = "{0:4d},".format(step)

    if j0 != control and status == 1:
        if overshootcond:
            sts = 3
        else:
            sts = 2
    else:
        sts = status

    message = ["Failed       ", " " * 13, " => re-Cycle ", "tol.Overshoot"]
    print(
        "|{0:4s}{1:1d} | {2:+4d}  => {3:+4d} | {4:.1e} ({5:2d}#) |{6:13s}|".format(
            stp, cycl, j0, control, fnorm, niterations, message[sts]
        )
    )


def errorcontrol():
    print("")
    print("ERROR. Control component changed in last cycle.")
    print("       Possible solution: Reduce stepwidth.")


def errorfinal():
    print("")
    print("ERROR. Numerical continuation stopped.")
