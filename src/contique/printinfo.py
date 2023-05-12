"""
contique: Numerical continuation of nonlinear equilibrium equations.
Andreas Dutzler, 2023
"""
import numpy as np


def header():
    print("|Step,C.| Control Component | Norm (Iter.#) | Message     |")
    print("|-------|-------------------|---------------|-------------|")


def cycle(step, cycl, control0, control, status, fnorm, niterations, overshootcond):
    if cycl > 1:
        stp = "     "
    else:
        stp = "{0:4d},".format(step)

    if not np.allclose(control0, control) and status == 1:
        if overshootcond:
            sts = 3
        else:
            sts = 2
    else:
        sts = status

    message = ["Failed       ", " " * 13, " => re-Cycle ", "tol.Overshoot"]
    sign0 = f"{control0[1]:+d}"[0]
    sign = f"{control[1]:+d}"[0]
    print(
        "|{0:4s}{1:1d} |{2:6d}{3:s}  =>{4:6d}{5:s} | {6:.1e} ({7:2d}#) |{8:13s}|".format(  # NOQA: E501
            stp,
            cycl,
            control0[0],
            sign0,
            control[0],
            sign,
            fnorm,
            niterations,
            message[sts],
        )
    )


def errorcontrol():
    print("")
    print("ERROR. Control component changed in last cycle.")
    print("       Possible solution: Reduce stepwidth.")


def errorfinal():
    print("")
    print("ERROR. Numerical continuation stopped.")
