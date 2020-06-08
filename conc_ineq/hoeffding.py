import math

import numpy as np


def hoeffding(n, delta, upper, lower):
    """
        Input:
            n: number of Xi's
            delta: error probability
            upper, lower:  lower <= Xi <= upper with probability 1

        Output:
            t: the number that satisfies the Hoeffding inequality:
                Pr(Sn >= t) <= {delta = exp(-t^2/{2n*sigma^2})}
                where Sn = X1 + ... + Xn.
    """
    tmp = -np.log(delta)
    half = (upper - lower) / 2
    return np.sqrt(2 * n * half ** 2 * tmp)


def adaptive_hoeffding(n, delta, upper, lower):
    """
        Adaptive Hoeffding-like Concentration Bounds
        as in Corollary 1 of Ermon 2016 NeurIPS paper.
        Valid for lower <= Xi <= upper.

        Input:
            n: number of Xi's
            delta: error probability
            upper, lower:  lower <= Xi <= upper with probability 1

        Output:
            The adaptive upper bound of Sn where Sn = X1 + ... + Xn.
    """
    b = -np.log(delta / 12.0) / 1.8
    return (upper - lower) * np.sqrt(0.6 * n * np.log(math.log(n, 1.1) + 1) + b * n)


def adaptive_hoeffding_seq(N, delta, upper, lower):
    """
        Adaptive Hoeffding-like Concentration Bounds
        as in Corollary 1 of Ermon 2016 NeurIPS paper.
        Valid for lower <= Xi <= upper.

        Input:
            N: number of Xi's
            delta: error probability
            upper, lower:  lower <= Xi <= upper with probability 1

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """
    b = -np.log(delta / 12.0) / 1.8
    n = np.arange(1, N + 1)
    return (upper - lower) * np.sqrt(
        0.6 * n * np.log(np.log(n) / np.log(1.1) + 1) + b * n
    )


def hrms_hoeffding_seq(N, delta, sigma=1.0):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, n in [N]
        using Eq 8 in Howard et al. 2018, where Xi is sigma sub-Gaussian.

        Input:
            N: number of Xi's
            delta: error probability
            sigma: subGaussian parameter. Xi is sigma-subGaussian

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """

    n = np.arange(1, N + 1)
    return (
        1.7 * sigma * np.sqrt(n * (np.log(np.log(2 * n)) + 0.72 * np.log(5.2 / delta)))
    )
