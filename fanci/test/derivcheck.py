# Derivcheck is robust and very sensitive tester for analytic derivatives.
# Copyright (C) 2017 Toon Verstraelen <Toon.Verstraelen@UGent.be>.
#
# This file is part of Derivcheck.
#
# Derivcheck is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# Derivcheck is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
# --
"""Robust and sensitive tester for first-order analytic partial derivatives."""


import numpy as np


__all__ = ["diff_ridders", "assert_deriv"]


def diff_ridders(function, origin, stepsize, con=1.4, safe=2.0, maxiter=15):
    """Estimate first-order derivative with Ridders' finite difference method.

    This implementation is based on the one from the book Numerical Recipes. The code
    is pythonized and no longer using fixed-size arrays. Also, the output of the function
    can be an array.

    Parameters
    ----------
    function : function
        The function to be differentiated.
    origin : float
        The point at which must be differentiated.
    stepsize : float
        The initial step size.
    con : float
        The rate at which the step size is decreased (contracted). Must be larger than
        one.
    safe : float
        The safety check used to terminate the algorithm. If Errors between successive
        orders become larger than ``safe`` times the error on the best estimate, the
        algorithm stop. This happens due to round-off errors.
    maxiter : int
        The maximum number of iterations, equals the maximum number of function calls and
        also the highest polynomial order in the Neville method.

    Returns
    -------
    estimate : float
        The best estimate of the first-order derivative.
    error : float
        The (optimistic) estimate of the error on the derivative.

    """
    if stepsize == 0.0:
        raise ValueError("stepsize must be nonzero.")
    if con <= 1.0:
        raise ValueError("con must be larger than one.")
    if safe <= 1.0:
        raise ValueError("safe must be larger than one.")

    con2 = con * con
    table = [
        [
            (np.asarray(function(origin + stepsize)) - np.asarray(function(origin - stepsize)))
            / (2.0 * stepsize)
        ]
    ]
    estimate = None
    error = None

    # Loop based on Neville's method.
    # Successive rows in the table will go to smaller stepsizes.
    # Successive columns in the table go to higher orders of extrapolation.
    for i in range(1, maxiter):
        # Reduce step size.
        stepsize /= con
        # First-order approximation at current step size.
        table.append(
            [
                (np.asarray(function(origin + stepsize)) - np.asarray(function(origin - stepsize)))
                / (2.0 * stepsize)
            ]
        )
        # Compute higher-orders
        fac = con2
        for j in range(1, i + 1):
            # Compute extrapolations of various orders, requiring no new
            # function evaluations. This is a recursion relation based on
            # Neville's method.
            table[i].append((table[i][j - 1] * fac - table[i - 1][j - 1]) / (fac - 1.0))
            fac = con2 * fac

            # The error strategy is to compare each new extrapolation to one
            # order lower, both at the present stepsize and the previous one:
            current_error = max(
                abs(table[i][j] - table[i][j - 1]).max(),
                abs(table[i][j] - table[i - 1][j - 1]).max(),
            )

            # If error has decreased, save the improved estimate.
            if error is None or current_error <= error:
                error = current_error
                estimate = table[i][j]

        # If the highest-order estimate is growing larger than the error on the best
        # estimate, the algorithm becomes numerically instable. Time to quit.
        if abs(table[i][i] - table[i - 1][i - 1]).max() >= safe * error:
            break
        i += 1
    return estimate, error


class OneDimWrapper:
    """Construct a function of a one-dimensional argument from an array function."""

    def __init__(self, function, origin, indices):
        """Initialize a OneDimWrapper object.

        Parameters
        ----------
        function : function
            The original function with an array argument.
        origin : np.ndarray
            The origin, corresponds to the one-dimensional argument equal to 0.
        indices : tuple
            The index of the matrix element of the input array to use

        """
        self.function = function
        self.origin = origin.copy()
        self.indices = indices

    def __call__(self, arg1):
        """Compute the one-dimensional function."""
        arg = self.origin.copy()
        arg[self.indices] += arg1
        return self.function(arg)


def assert_deriv(function, gradient, origin, widths=0.1, output_mask=None, rtol=1e-5, atol=1e-8):
    """Test the gradient of a function.

    Parameters
    ----------
    function : function
        The function whose derivatives must be tested, takes one argument, which may be
        a scalar or an array with shape ``shape_in``. It may also return an array with
        shape ``shape_out``.
    gradient : function
        Computes the gradient of the function, to be tested. It takes one argument, same
        type as ``function``. The return value is an array with shape ``shape_out +
        shape_in``.
    origin : np.ndarray
        The point at which the derivatives are computed.
    widths : float or np.ndarray
        The initial (maximal) step size for the finite difference method. Do not take a
        value that is too small. When an array is given, each matrix element of the input
        of the function gets a different step size. When a matrix element is set to zero,
        the derivative towards that element is not tested. The function will not be
        sampled beyond [origin-widths, origin+widths].
    output_mask : np.ndarray or None
        This option is useful when the function returns an array output: it allows the
        caller to select which components of the output need to be tested. When not given,
        all components are tested.
    rtol : float
        The allowed relative error on the derivative.
    atol : float
        The allowed absolute error on the derivative.

    Returns
    -------
    numtested : int
        The number of derivatives tested.

    Raises
    ------
    AssertionError when the error on the derivative is too large.

    """
    # Make sure origin is always an array object.
    origin = np.asarray(origin)

    # Compute the gradient and give it the 1D or 2D shape. The first index is a raveled
    # output index.
    gradient = np.asarray(gradient(origin))
    if output_mask is not None:
        gradient = gradient[output_mask]
    if origin.ndim == 0:
        gradient = gradient.ravel()
    else:
        gradient = gradient.reshape(-1, origin.size)

    # Flat loop ofer all elements of the input array
    numtested = 0
    for iaxis in range(origin.size):
        # Get the corresponding input array indices.
        if origin.ndim == 0:
            indices = ()
        else:
            indices = np.unravel_index(iaxis, origin.shape)

        # Determine the step size
        if isinstance(widths, float):
            stepsize = widths
        else:
            stepsize = widths[indices]

        # If needed, test this component
        if stepsize > 0:
            # Make a function of only the selected input array element.
            wrapper = OneDimWrapper(function, origin, indices)
            # Compute the numerical derivative of this function and an error estimate.
            deriv_approx, deriv_error = diff_ridders(wrapper, 0.0, stepsize)
            # Get the corresponding analytic derivative.
            deriv = gradient[..., iaxis]
            # Make sure the error on the derivative is smaller than the requested
            # thresholds.
            if deriv_error >= atol and deriv_error >= rtol * abs(deriv).max():
                raise FloatingPointError(
                    "Inaccurate estimate of the derivative for " "index={}.".format(indices)
                )
            # Flatten the array with numerical derivatives.
            if output_mask is None:
                deriv_approx = deriv_approx.ravel()
            else:
                deriv_approx = deriv_approx[output_mask]
            # Compare
            err_msg = "derivative toward {} x=analytic y=numeric stepsize={:g}".format(
                indices, stepsize
            )
            np.testing.assert_allclose(deriv, deriv_approx, rtol, atol, err_msg=err_msg)
            numtested += deriv.size
    return numtested
