"""
contique: Numerical continuation of nonlinear equilibrium equations.
Andreas Dutzler, 2023
"""
import numpy as np
from scipy import sparse

from .helpers import argparser


class NewtonResult:
    """Class for handling the results of a Newton-Rhapson solution.

    This class has several public attribues.

    Attributes
    ----------
    success : bool
        flag for the converged solution
    message : str
        message for the state
    status : int
        integer representig the status of the solution (0 if not converged, 1 if
        converged).
    niterations : int
        number of performed iterations
    x : ndarray
        1d-array containing the unknows
    fun : function
        function returning the equilibrium equations
    jac : function
        function returning the jacobian of the equilibrium equations

    """

    def __init__(self, fun, x0, jac, args):
        """Initialize an Instance of a NewtonResult.

        Parameters
        ----------
        fun : function
            function returning the equilibrium equations
        x0 : ndarray
            1d-array containing the initial unknows
        jac : function, optional
            function returning the jacobian of the equilibrium equations
        args : tuple, optional
            Optional tuple of arguments which are passed to the function. Eeven if only
            one argument is passed, it has to be encapsulated in a tuple (default is
            (None,)).

        """
        self.success = False
        self.message = "not started"
        self.status = 0
        self.niterations = 0
        self.x = x0.copy()
        self.fun = argparser(fun)(self.x, *args)

        if jac is not None:
            self.jac = argparser(jac)(self.x, *args)


def newtonrhapson(fun, x0, jac, args=(None,), maxiter=8, tol=1e-8, solve=None):
    """A simple n-dimensional Newton-Rhapson solver.

    Parameters
    ----------
    fun : function
        function in terms of unknows x and optional args which returns the
        equilibrium equations.
    x0 : ndarray
        1d-array with initial values of unknows x
    jac : function
        jacobian of fun w.r.t. the unknows x
    args : tuple, optional
        Optional tuple of arguments which are passed to the function. Eeven if only
        one argument is passed, it has to be encapsulated in a tuple (default is
        (None,)).
    maxiter : int, optional
        maximum number of iterations (default is 8)
    tol : float, optional
        tolerated residual of the norm of the equilibrium equation (default is 1e-8)

    Returns
    -------
    res : NewtonResult
        Instance of NewtonResult with res.x being the final unknowns

    """

    # init result object with initial function evaluation
    res = NewtonResult(fun, x0, None, args)

    # iteration loop
    for res.niterations in range(1, 1 + maxiter):

        # calculate jacobian at x
        res.jac = argparser(jac)(res.x, *args)

        # set solver according to dense or sparse jacobian
        if solve is None:
            if sparse.issparse(res.jac):
                solve = sparse.linalg.spsolve
            else:
                solve = np.linalg.solve

        # solve linear equation system
        try:
            res.x += solve(res.jac, -res.fun)
        except:  # NOQA: E722
            res.x *= np.nan

        # calculate function at updated x
        res.fun = argparser(fun)(res.x, *args)

        # convergence check
        if np.linalg.norm(res.fun) < tol:
            res.success = True
            res.status = 1
            res.message = "Solution converged in {0:2d} Iteration".format(
                res.niterations
            )
            if res.niterations > 1:
                res.message = res.message + "s"

            break

    # check if newton process failed
    if not res.success:
        if maxiter == 1:
            res.message = " ".join(
                [
                    "Calculated linear solution",
                    "because of input parameter `maxiter=1` (not converged).",
                ]
            )
        else:
            res.message = "Newton-R. process failed."

    return res
