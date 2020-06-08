import math

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from confseq import boundaries


def bernstein(n, delta, A, B):
    """
        Input:
            n: number of Xi's
            delta: error probability
            A: std deviation of Xi's
            B: Xi is upper bounded by B with probability 1.

        Output: n * t
                where t is the number that satisfies the Bernstein inequality:
                Pr( Sn >= nt ) <= {delta = exp(-nt^2/{2A^2 + 2Bt/3})}
    """
    tmp = -np.log(delta)
    return np.sqrt(2 * n * A ** 2 * tmp + (B * tmp) ** 2 / 9) + B * tmp / 3


def empirical_bernstein(n, delta, A, B, eta=1.1, power=1.1):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, using the empirical
        Bernstein bound (Empirical Bernstein Stopping, Mnih et al.  ICML 2008) , where
        the standard deviation and upper bound of r.v. Xi is known to be A and B.

        Input:
            n: number of Xi's
            delta: error probability
            A: std deviation of Xi's
            B: Xi is upper bounded by B with probability 1.
            eta, power: parameters for stitching.

        Output:
            The adaptive upper bound of Sn
    """
    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    kmax = np.ceil(math.log(n, eta))
    k = np.max([0, kmax - 1])
    delta_k = delta / h(k)
    x = np.log(1.5 / delta_k)
    return A * np.sqrt(2 * x * n * eta) + 3 * B * eta * x


def empirical_bernstein_seq_known_var(N, delta, A, B, eta=1.1, power=1.1):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, n in [N] using the
        empirical Bernstein bound (Empirical Bernstein Stopping, Mnih et al. ICML 2008),
        where the standard deviation and uper bound of r.v. Xi is known to be A and B.

        Input:
            N: number of Xi's
            delta: error probability
            A: std deviation of Xi's
            B: Xi is upper bounded by B with probability 1.
            eta, power: parameters for stitching.

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """
    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    kmax = np.ceil(math.log(N, eta))

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    f = np.zeros(N)
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_k = delta / h(k)
            x = np.log(1.5 / delta_k)
            n = np.arange(nmin, min(nmax, N) + 1)
            f[n - 1] = A * np.sqrt(2 * x * n * eta) + 3 * B * eta * x
    assert len(f) == N
    return f


def empirical_bernstein_seq(X, delta, B, std_upper_bound, eta=1.1, power=1.1):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, n in [N] using the
        empirical Bernstein bound (Empirical Bernstein Stopping, Mnih et al. ICML 2008),
        where the data is assumed to be generated on the fly.

        Input:
            X: An array of Xi's
            delta: error probability
            B: Xi is upper bounded by B with probability 1.
            std_upper_bound: The upper bound of the standard deviation of Xi.
            eta, power: parameters for stitching.

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """
    N = X.size

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    kmax = np.ceil(math.log(N, eta))

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    f = np.zeros(N)
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_k = delta / h(k)
            x = np.log(1.5 / delta_k)
            n = np.arange(nmin, min(nmax, N) + 1)
            A = np.array([min(np.std(X[:ii]), std_upper_bound) for ii in n])
            f[n - 1] = A * np.sqrt(2 * x * n * eta) + 3 * B * eta * x
    assert len(f) == N
    return f


def hrms_bernstein_seq(X, delta, std_upper_bound):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, n in [N]
        using Eq 20 in Howard et al. 2018, where the data is assumed
        to be generated on the fly.

        The difference between Howard's method and the one used in
        empirical_bernstein_seq is they have different ways of stitching.

        Input:
            X: An array of Xi's
            delta: error probability
            std_upper_bound: The upper bound of the standard deviation of Xi.

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """

    N = X.size
    Xmean = np.cumsum(X) / np.arange(1, N + 1)
    lagged_means = np.roll(Xmean, 1)
    diff = X - lagged_means
    diff[0] = std_upper_bound
    V = np.minimum(np.cumsum(diff ** 2), std_upper_bound ** 2 * np.arange(1, N + 1))

    tmp = np.log(np.log(2 * np.maximum(V, 1.0)))
    tmp[np.isnan(tmp)] = 0

    return 1.7 * np.sqrt(V * (tmp + 3.8)) + 3.4 * tmp + 13


def hrms_bernstein_gamma_seq(X, delta, L, U, propensity=0.5, v_opt=500, alpha_opt=0.05):
    """
        Compute the uniform upper bound for Sn = X1 + ... + Xn, n in [N]
        using in Howard et al. 2018, using empirical Bernstein with
        gamma exponential mixture.

        Input:
            X: An array of Xi's
            delta: error probability
            L, U: Pr(L <= X <= U) = 1
            propensity, alpha_opt, v_opt: parameters for the gamma exponential mixture.
                                          The upper bound will be tightest at n=v_opt.

        Output:
            A sequence of adaptive upper bounds for S1, S2, ..., SN.
    """
    std_upper_bound = (U - L) / 2.0
    N = X.size
    Xmean = np.cumsum(X) / np.arange(1, N + 1)
    lagged_means = np.roll(Xmean, 1)
    diff = X - lagged_means
    diff[0] = std_upper_bound
    V = np.minimum(np.cumsum(diff ** 2), std_upper_bound ** 2 * np.arange(1, N + 1))

    p_min = min(propensity, 1 - propensity)
    support_diameter = U - L
    c = 2 * support_diameter / p_min
    return boundaries.gamma_exponential_mixture_bound(
        V, delta / 2, v_opt, c, alpha_opt=alpha_opt / 2
    )
