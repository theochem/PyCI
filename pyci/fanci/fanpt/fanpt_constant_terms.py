r"""Class that generates and contains the constant terms of the FANPT system of equations."""

from math import factorial

import numpy as np

from .base_fanpt_container import FANPTContainer


class FANPTConstantTerms:
    r"""Generates and contains the constant terms of the FANPT system of equations.

    If the order is 1:
    -dG_n/dl

    If the energy is not an active parameter:
    -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}

    If the energy is an active parameter:
    linear_term = t1 + t2
    t1 = -N * sum_k {d^2G_n/dldp_k * d^(N-1)p_k/dl^(N-1)}
    t2 = - sum_k{d^2G_n/dEdp_k * sum_m{C(N,m) * d^mE/dl^m * d^(N-m)p_k/dl^(N-m)}}

    Notation
    N: order
    sum_k: sum over all active wfn parameters.
    sum_m: sum from 1 to N-1.
    C(N,m): N-choose-m (binomial coefficient). Calculated using math.factorial because
            math.comb is only available starting at Python 3.8.

    Attributes
    ----------
    fanpt_container : FANPTContainer
        Object containing the FANPT matrices and vectors.
    order : int
        Order of the current FANPT iteration.
    previous_responses : np.ndarray
        Previous responses of the FANPT calculations up to order = order -1

    Methods
    -------
    __init__(self, fanpt_container, order=1, previous_responses=None)
        Initialize the constant terms.
    assign_fanpt_container(self, fanpt_container)
        Assign the FANPT container.
    assign_order(self, order)
        Assign the order.
    assign_previous_responses(self, previous_responses)
        Assign the previous responses.
    gen_constant_terms(self)
        Generate the constant terms.
    """

    def __init__(self, fanpt_container, order=1, previous_responses=[]):
        r"""Initialize the constant terms.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            Object containing the FANPT matrices and vectors.
        order : int
            Order of the current FANPT iteration.
        previous_responses : np.ndarray
            Previous responses of the FANPT calculations up to order = order -1
        """
        self.assign_fanpt_container(fanpt_container=fanpt_container)
        self.assign_order(order=order)
        self.assign_previous_responses(previous_responses=previous_responses)
        self.gen_constant_terms()

    def assign_fanpt_container(self, fanpt_container):
        r"""Assign the FANPT container.

        Parameters
        ----------
        fanpt_container : FANPTContainer
            FANPTContainer object that contains all the matrices and vectors required to perform
            the FANPT calculation.

        Raises
        ------
        TypeError
            If fanpt_container is not a child of FANPTContainer.
        """
        if not isinstance(fanpt_container, FANPTContainer):
            raise TypeError("fanpt_container must be a child of FANPTContainer")
        self.fanpt_container = fanpt_container

    def assign_order(self, order):
        r"""Assign the order.

        Parameters
        ----------
        order : int
            Order of the current FANPT iteration.

        Raises
        ------
        TypeError
            If order is not an int.
        ValueError
            If order is negative.
        """
        if not isinstance(order, int):
            raise TypeError("order must be an integer.")
        if order < 0:
            raise ValueError("order must be non-negative")
        self.order = order

    def assign_previous_responses(self, previous_responses):
        r"""Assign the previous responses.

        Parameters
        ----------
        previous_responses : np.ndarray
            Previous responses of the FANPT calculations up to order = order -1.

        Raises
        ------
        TypeError
            If previous_responses is not a numpy array.
            If the elements of previous_responses are not numpy arrays.
        ValueError
            If previous_responses is None and order is not 1.
            If the shape of previous_responses is not equal to (order - 1, nactive).
        """
        if self.order == 1:
            self.previous_responses = previous_responses
        else:
            if not isinstance(previous_responses, np.ndarray):
                raise TypeError("previous_responses must be a numpy array.")
            if not all([isinstance(response, np.ndarray) for response in previous_responses]):
                raise TypeError("The elements of previous_responses must be numpy arrays.")
            if previous_responses.shape != (self.order - 1, self.fanpt_container.nactive):
                raise ValueError(
                    "The shape of previous_responses must be ({}, {}).".format(
                        self.order - 1, self.fanpt_container.nactive
                    )
                )
            self.previous_responses = previous_responses

    def gen_constant_terms(self):
        r"""Generate the constant terms.

        Returns
        -------
        constant_terms : np.ndarray
            Constant terms of the FANPT linear system of equations.
            numpy array with shape (nequation,).
        """
        if self.order == 1:
            constant_terms = -self.fanpt_container.d_g_lambda
        else:
            if self.fanpt_container.active_energy:
                r_vector = np.zeros(self.fanpt_container.nactive - 1)
                for o in range(1, self.order):
                    comb = factorial(self.order) / (factorial(o) * factorial(self.order - o))
                    r_vector += (
                        comb
                        * self.previous_responses[o - 1][-1]
                        * self.previous_responses[self.order - o - 1][:-1]
                    )
                constant_terms = -self.order * np.dot(
                    self.fanpt_container.d2_g_lambda_wfnparams, self.previous_responses[-1][:-1]
                ) - np.dot(self.fanpt_container.d2_g_e_wfnparams, r_vector)
            else:
                constant_terms = -self.order * np.dot(
                    self.fanpt_container.d2_g_lambda_wfnparams, self.previous_responses[-1]
                )
        self.constant_terms = constant_terms
