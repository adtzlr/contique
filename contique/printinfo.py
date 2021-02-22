import numpy as np


def stepheader(step):
    print("\nBegin of Step %d" % step, "\n" + "=" * 68 + "\n")
    print("| Cycle | converged in " + 23 * " " + "| control component  |")
    print("|" + "-" * 7 + "|" + "-" * 37 + "|" + "-" * 20 + "|")


def cycle(cycl, msg, j0, control):
    print(
        "| #{0:4d} | {1:s} | from {2:+4d} to {3:+4d}  |".format(cycl, msg, j0, control)
    )


def stepfinal(x, control, fun):
    print("")
    print("*final lpf value     = {: .3e}".format(x[-1]))
    print("*final control value = {: .3e}".format(x[abs(control) - 1]))
    print("*final equilibrium   = {: .3e} (norm)".format(np.linalg.norm(fun)))
