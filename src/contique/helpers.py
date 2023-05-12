"""
contique: Numerical continuation of nonlinear equilibrium equations.
Andreas Dutzler, 2023
"""

import numpy as np


def needle(component: int, length: int):
    """Return an array of zeros with a needle at a given ``component`` and
    ``length``. This array is used to slice out the j-th component of another
    1d-array.

    ..  code-block::

        x_j = needle(component, length).dot(x)


    Furthermore, the derivative of this  equation w.r.t. ``x`` results in
    ``needle(component, length)``.

    Parameters
    ----------
    component : int
        Component at which the needle should be positioned.
    length : int
        Length of the output array.

    Returns
    -------
    ndarray
        1d-array with a needle at item ``component`` and zero for all other
        items.

    Examples
    --------
    A needle vector with ``length = 9`` and a needle at ``component=5``.

    >>> needle(5, 9)
    array([0, 0, 0, 0, 0, 1, 0, 0, 0])

    """

    # init a 1d-array filled with zeros
    n = np.zeros(length, dtype=int)

    # insert needle
    n[component] = 1

    return n


def control(x):
    """Obtain the index and the sign of the greatest absolute value of a
    1d-array. The returned integer and the sign are taken from
    the greatest value.

    Parameters
    ----------
    x : ndarray
        1d-array

    Returns
    -------
    int
        Position which contains the greatest absolute value of `x`.
    int
        Sign of the greatest value.
    """

    # 0-indexed position
    component = abs(x).argmax()

    return component, int(np.sign(x[component]))


def argparser(fun):
    "Function decorator for the handling of function arguments."

    def inner(x, *args, **kwargs):
        "Pass `*args` and `**kwargs` to a function if they are not None."
        no_args = (len(args) == 1 and args[0] is None) or not bool(args)
        no_kwargs = not bool(kwargs)

        if no_args and no_kwargs:
            f = fun(x)
        elif not no_args and no_kwargs:
            f = fun(x, *args)
        elif no_args and not no_kwargs:
            f = fun(x, **kwargs)
        else:
            f = fun(x, *args, **kwargs)
        return f

    return inner


def argparser2(fun):
    """Function decorator for the handling of function arguments
    with 2 primary arguments followed by other args."""

    def inner2(x, lpf, *args, **kwargs):
        "Pass `*args` and `**kwargs` to a function if they are not None."
        no_args = (len(args) == 1 and args[0] is None) or not bool(args)
        no_kwargs = not bool(kwargs)

        if no_args and no_kwargs:
            f = fun(x, lpf)
        elif not no_args and no_kwargs:
            f = fun(x, lpf, *args)
        elif no_args and not no_kwargs:
            f = fun(x, lpf, **kwargs)
        else:
            f = fun(x, lpf, *args, **kwargs)
        return f

    return inner2
