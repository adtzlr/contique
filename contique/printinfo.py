import numpy as np


def header():
    print("| Step (Cycle) | Control Comp. | Equili. | Status  |")
    print("|--------------|---------------|---------|---------|")


def cycle(step, cycl, j0, control, status, fnorm):
    if cycl > 1:
        stp = "    "
    else:
        stp = "{0:4d}".format(step)

    if j0 != control and status == 1:
        sts = 2
    else:
        sts = status

    message = [" Failed", "Success", "Recycle"]
    print(
        "| {0:4s}     ({1:1d}) | {2:+4d}  => {3:+4d} | {4:.1e} | {5:s} |".format(
            stp, cycl, j0, control, fnorm, message[sts]
        )
    )


def errorcontrol():
    print("")
    print("ERROR. Control component changed in last cycle.")
    print("       Possible solution: Reduce stepwidth.")


def errorfinal():
    print("")
    print("ERROR. Numerical continuation stopped.")
