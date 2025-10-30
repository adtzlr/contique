"""
contique: Numerical continuation of nonlinear equilibrium equations.
"""

import numpy as np


def one_hot(component: int, length: int) -> np.ndarray:
    """Return an array with a given length, which contains zeros and a single item at
    index ``component`` with value one. This array is used to slice out the one-hot
    j-th component of another 1d-array.

    Parameters
    ----------
    component : int
        Component at which the value of 1 should be placed.
    length : int
        Length of the output array.

    Returns
    -------
    ndarray
        1d-array with a value of one at index ``component`` and zeros for all other
        items.

    Notes
    -----
    The one-hot array can be used to extract the j-th component of another 1d-array
    ``x`` of the same length via the dot product.

    ..  code-block::

        x_j = one_hot(component, length).dot(x)

    The derivative of this  equation w.r.t. ``x`` results in
    ``one_hot(component, length)``.

    Examples
    --------
    A one-hot vector with ``length = 9`` and ``component=5``.

    >>> contique.one_hot(5, 9)
    array([0, 0, 0, 0, 0, 1, 0, 0, 0])

    """

    # init a 1d-array filled with zeros
    n = np.zeros(length, dtype=int)

    # insert one-hot component
    n[component] = 1

    return n


def control(x: np.ndarray) -> tuple[int, int]:
    """Obtain the index and the sign of the greatest absolute value of a 1d-array. The
    returned integer and the sign are taken from the greatest value.

    Parameters
    ----------
    x : ndarray
        Input 1d-array

    Returns
    -------
    int
        Index which contains the greatest absolute value of `x`.
    int
        Sign of the greatest value.
    """

    # 0-indexed position
    idx = abs(x).argmax()

    return idx, int(np.sign(x[idx]))


def argparser(fun: callable) -> callable:
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


def argparser2(fun: callable) -> callable:
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
