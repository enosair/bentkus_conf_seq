import math

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats


class QFunc(object):
    """
        Class for computing the q(delta; n, A, B) function in Bentkus ConfSeq paper.
    """

    def __init__(self, n, A=0.5, B=1.0):
        self.n = n
        self.A = A
        self.B = B
        self.bias = A ** 2 / (A ** 2 + B ** 2)
        self.nb = n * self.bias
        self.mass = scipy.stats.binom.pmf(range(n + 1), n, self.bias)

    def p(self, k):
        assert k <= self.n
        return np.sum(self.mass[k:])

    def e(self, k):
        assert k <= self.n
        return np.sum(np.arange(k, self.n + 1) * self.mass[k:])

    def v(self, k):
        assert k <= self.n
        return np.sum(np.arange(k, self.n + 1) ** 2 * self.mass[k:])

    def P2(self, x):
        """
            Function P2(x;Zn) where Zn is a Binom r.v. with bias = self.bias.
        """
        if x <= self.nb:
            return 1.0
        elif x >= self.n:
            return self.bias ** self.n
        elif x <= self.v(0) / self.e(0):
            return (
                self.nb
                * (1 - self.bias)
                / ((x - self.nb) ** 2 + self.nb * (1 - self.bias))
            )
        else:
            for k in range(1, self.n):
                pp = self.p(k)
                ee = self.e(k)
                vv = self.v(k)
                if x <= self.F(k, pp, ee, vv):
                    return (vv * pp - ee ** 2) / (x ** 2 * pp - 2 * x * ee + vv)

        print("Did not find a solution.")
        raise

    def F(self, h, pp=None, ee=None, vv=None):
        """
            Function F(h) = E[Zn(Zn-h)]/E[Zn-h] as in Appendix C.
            Zn is a Binom r.v. with bias = self.bias.
        """
        if h <= 0:
            return self.nb + self.nb * (1 - self.bias) / (self.nb - h)
        elif h >= self.n - 1:
            return self.n
        else:
            k = int(np.ceil(h))
            if pp is None:
                pp = self.p(k)
                ee = self.e(k)
                vv = self.v(k)
            return (vv - h * ee) / (ee - h * pp)

    def delta_func(self, k, pp=None, ee=None, vv=None):
        """
            Compute delta_k = P2(ratio_k;Zn).

            P2 is a piecewise function, and ratio_k's
            are the changing points. In particular,
            ratio_k = F(k).
        """
        if pp is None:
            pp = self.p(k)
            ee = self.e(k)
            vv = self.v(k)
        x = self.F(k, pp, ee, vv)
        return (vv * pp - ee ** 2) / (x ** 2 * pp - 2 * x * ee + vv)

    def __call__(self, delta):
        """
            Evaluate mu = q(delta; n, A, B), which is the value such that
            delta = P2(mu, sum of Gi).

            The idea is to transform the problem into searching for mu1
            such that delta = P2(mu1; Zn), then compute compute mu from mu1.

            Since P2(mu1; Zn) is a piecewise function, we can find the range
            where delta falls into, and invert the function of the corresponding
            piece.
        """
        if delta >= 1:
            mu1 = -np.inf
        elif delta <= self.bias ** self.n:
            mu1 = self.n + 1e-8
        else:
            success = False

            #  min(left, ratio(left)) <= mu1 <= right <= ratio(right)
            left = int(scipy.stats.binom.ppf(1 - delta, self.n, self.bias))
            right = int(
                scipy.stats.binom.ppf(1 - 2 * delta / np.exp(2), self.n, self.bias)
            )
            delta_left = self.delta_func(left)

            #  delta_left is evaluated at ratio_left. If delta >= delta_left,
            #  then mu1 <= ratio_left as P2 function is monotonically decreasing.
            #  Search through pieces with index 0, ... left
            #  search_left and search_right is for k - 1: which is the index of delta_u
            #  search_left == -1 corresponds to the case where delta is between [v0/e0, 1]
            if delta >= delta_left:
                search_left = -1
                search_right = min(self.n - 1, left)
            # Otherwise search through pieces with index left + 1, ..., right
            else:
                search_left = left
                search_right = min(self.n - 1, right + 1)

            # delta_u is indexed by k-1. k only goes up to
            # n-1. In this case, the right boundary should be the lowest delta possible.
            if search_right == self.n - 1:
                delta_b = self.bias ** self.n
            # when k - 1 <= n-2, proceed with delta_b = delta_(k)
            else:
                delta_b = self.delta_func(search_right + 1)

            # k_1 is the index of delta_u
            for k_1 in range(search_right, search_left - 1, -1):
                if k_1 == -1:
                    delta_u = 1.0
                else:
                    if k_1 == search_right:
                        pp = self.p(k_1)
                        ee = self.e(k_1)
                        vv = self.v(k_1)
                    else:
                        pp = pp + self.mass[k_1]
                        ee = ee + k_1 * self.mass[k_1]
                        vv = vv + k_1 ** 2 * self.mass[k_1]

                    delta_u = self.delta_func(k_1, pp, ee, vv)

                if delta_b < delta and delta <= delta_u:
                    if k_1 == -1:
                        mu1 = self.nb + np.sqrt(
                            self.nb * (1 - self.nb) * (1 - delta) / delta
                        )
                    else:
                        k = k_1 + 1
                        pp = self.p(k)
                        ee = self.e(k)
                        vv = self.v(k)
                        a = pp
                        b = -2 * ee
                        c = vv + (ee ** 2 - vv * pp) / delta
                        mu1 = (-b + np.sqrt(b ** 2 - 4.0 * a * c)) / 2.0 / a
                    success = True
                    break
                else:
                    delta_b = delta_u

            # Slow version: search from up to down, delta_b is with index k.
            # delta_u = 1.0
            # for k in range(self.n + 1):
            #     if k == 0:
            #         pp = self.p(k)
            #         ee = self.e(k)
            #         vv = self.v(k)
            #     else:
            #         pp = pp - self.mass[k - 1]
            #         ee = ee - (k - 1) * self.mass[k - 1]
            #         vv = vv - (k - 1) ** 2 * self.mass[k - 1]

            #     delta_b = self.delta_func(k, pp, ee, vv)
            #     print("U: {}, B: {}, delta: {}".format(delta_u, delta_b, delta))
            #     if delta_b < delta and delta <= delta_u:
            #         if k == 0:
            #             mu1 = self.nb + np.sqrt(
            #                 self.nb * (1 - self.nb) * (1 - delta) / delta
            #             )
            #         else:
            #             a = pp
            #             b = -2 * ee
            #             c = vv + (ee ** 2 - vv * pp) / delta
            #             mu1 = (-b + np.sqrt(b ** 2 - 4.0 * a * c)) / 2.0 / a
            #         success = True
            #         print("k: {}".format(k))
            #         break
            #     else:
            #         delta_u = delta_b

            if not success:
                print("Didn't find the correct bin for delta")
                raise
        return (mu1 * (self.A ** 2 + self.B ** 2) - self.n * self.A ** 2) / self.B


def bentkus(n, delta, A, B):
    """
        Theorem 1 of Bentkus ConfSeq paper.

        Input:
            n: number of Xi's
            delta: error probability
            A: the upper bound of the variance of Xi.
            B: Xi upper bounded by B with probability 1.

        Output:
            The pointwise upper bound for Sn:
            Pr( Sn >= q_func ) <= delta.
    """

    q_func = QFunc(n=n, A=A, B=B)
    return q_func(delta)


def adaptive_bentkus_seq_known_var(N, delta, A, B, eta=1.1, power=1.1):
    """
        Theorem 2 of Bentkus ConfSeq paper.

        Input:
            N: max number of Xi's
            delta: error probability
            A: the upper bound of the variance of Xi.
            B: Xi upper bounded by B with probability 1.

        Output:
            A sequence of adaptive upper bounds for
               sum_{i=1}^n X_i, for n = 1, 2,..., N.
    """

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return ((k + 1) ** power) * c

    kmax = np.ceil(math.log(N, eta))
    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    # all the n's inside the interval will get the same upper bound
    f = np.zeros(N)
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            q_func = QFunc(n=nmax, A=A, B=B)
            val = q_func(delta / h(k))
            n = np.arange(nmin, min(nmax, N) + 1)
            f[n - 1] = val
    return f


def adaptive_bentkus_seq_mean_zero(
    X, delta, B, eta=1.1, power=1.1, std_upper_bound=None
):
    """
        Theorem 3 of Bentkus ConfSeq paper.

        Input:
            X: An array of mean zero r.v. Xi's
            delta: error probability
            B: Pr(-B <= Xi <= B) = 1.
            eta, power: parameters for stitching.
            std_upper_bound: The upper bound of the standard deviation of Xi.

        Output:
            A sequence of upper bounds of sum_{i=1}^n X_i, for n = 1, 2,..., N
    """
    if not std_upper_bound:
        std_upper_bound = B
    else:
        std_upper_bound = min(B, std_upper_bound)

    delta_1 = 2 * delta / 3  # for q function
    delta_2 = delta - delta_1  # for the upper bound of std

    N = X.size

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    kmax = np.ceil(math.log(N, eta))

    def g_function(n, nmax, delta_2_k):
        quantitle = scipy.stats.norm.ppf(1 - 2 * delta_2_k / np.exp(2))
        return np.sqrt(np.floor(nmax / 2.0)) * 2 * B * quantitle / (n * 2 * np.sqrt(2))

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    f = np.zeros(N)
    Abar_min = std_upper_bound
    sq_diff = 0.0
    Ahat = np.inf
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_1_k = delta_1 / h(k)
            delta_2_k = delta_2 / h(k)
            for n in np.arange(nmin, min(nmax, N) + 1):
                if n == 1:
                    Abar = std_upper_bound
                    g = 0.0
                else:
                    if n % 2 == 0:
                        sq_diff += (X[n - 1] - X[n - 2]) ** 2 / 2.0
                        Ahat = np.sqrt(sq_diff / np.floor(n / 2.0))
                    g = g_function(n, nmax, delta_2_k)
                    Abar = np.sqrt(Ahat ** 2 + g ** 2) + g
                Abar_min = min(Abar, Abar_min)
                q_func = QFunc(n=nmax, A=Abar_min, B=B)
                val = q_func(delta_1_k / 2.0)
                f[n - 1] = val

    print("final Abar_min: {}".format(Abar_min))
    return f


def adaptive_bentkus_seq(Y, delta, L, U, eta=1.1, power=1.1):
    """
        Theorem 4 of Bentkus ConfSeq paper.

        Input:
            Y: An array of bounded r.v. Yi's with mean mu
            delta: error probability
            L, U: Pr(L <= Yi <= U) = 1
            eta, power: parameters for stitching.

        Output:
            mu_low, mu_up: (1-delta) confidence sequence of mu
            Sn_low, Sn_up: (1-delta) confidence sequence of S1, S2, ...
    """
    std_upper_bound = (U - L) / 2.0

    delta_1 = 2 * delta / 3  # for q function
    delta_2 = delta - delta_1  # for the upper bound of std

    N = Y.size

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    def g_function(n, nmax, delta_2_k):
        quantitle = scipy.stats.norm.ppf(1 - 2 * delta_2_k / np.exp(2))
        return (
            np.sqrt(np.floor(nmax / 2.0)) * (U - L) * quantitle / (n * 2 * np.sqrt(2))
        )

    # prepare return values
    mu_up = np.zeros(N)
    mu_low = np.zeros(N)
    Sn_up = np.zeros(N)
    Sn_low = np.zeros(N)

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    sq_diff = 0.0
    Ahat = np.inf
    Ymean = np.cumsum(Y) / np.arange(1, N + 1)
    Abar_min_prev = std_upper_bound

    kmax = np.ceil(math.log(N, eta))
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_1_k = delta_1 / h(k)
            delta_2_k = delta_2 / h(k)

            for n in np.arange(nmin, min(nmax, N) + 1):

                if n == 1:
                    Abar = std_upper_bound
                    g = 0.0
                else:
                    if n % 2 == 0:
                        sq_diff += (Y[n - 1] - Y[n - 2]) ** 2 / 2.0
                        Ahat = np.sqrt(sq_diff / np.floor(n / 2.0))
                    g = g_function(n, nmax, delta_2_k)
                    Abar = np.sqrt(Ahat ** 2 + g ** 2) + g

                Abar_min = min(Abar, Abar_min_prev)

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_up[n - 2] != mu_up[n - 3]
                ):
                    q_up_func = QFunc(
                        n=nmax, A=Abar_min, B=mu_up[n - 2] - L if n >= 2 else U - L
                    )
                    Sn_low[n - 1] = -q_up_func(delta_1_k / 2.0)
                else:
                    Sn_low[n - 1] = Sn_low[n - 2]
                mu_up[n - 1] = min(
                    Ymean[n - 1] - Sn_low[n - 1] / n, mu_up[n - 2] if n >= 2 else U
                )

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_low[n - 2] != mu_low[n - 3]
                ):
                    q_low_func = QFunc(
                        n=nmax, A=Abar_min, B=U - mu_low[n - 2] if n >= 2 else U - L
                    )
                    Sn_up[n - 1] = q_low_func(delta_1_k / 2.0)
                else:
                    Sn_up[n - 1] = Sn_up[n - 2]

                mu_low[n - 1] = max(
                    Ymean[n - 1] - Sn_up[n - 1] / n, mu_low[n - 2] if n >= 2 else L
                )

                Abar_min_prev = Abar_min

    # print("final Abar_min: {}".format(Abar_min))
    return mu_low, mu_up, Sn_low, Sn_up


def adaptive_bentkus_seq_untruncated(Y, delta, L, U, eta=1.1, power=1.1):
    """
        The version output upper and lower bounds that are not cumulative
        minimum or maximum, to be used for the best arm bandit algorithm.

        Input:
            Y: An array of bounded r.v. Yi's with mean mu
            delta: error probability
            L, U: Pr(L <= Yi <= U) = 1
            eta, power: parameters for stitching.

        Output:
            mu_low, mu_up: (1-delta) confidence sequence of mu (untruncated)
            Sn_low, Sn_up: (1-delta) confidence sequence of S1, S2, ...
    """
    std_upper_bound = (U - L) / 2.0

    delta_1 = 2 * delta / 3  # for q function
    delta_2 = delta - delta_1  # for the upper bound of std

    N = Y.size

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    def g_function(n, nmax, delta_2_k):
        quantitle = scipy.stats.norm.ppf(1 - 2 * delta_2_k / np.exp(2))
        return (
            np.sqrt(np.floor(nmax / 2.0)) * (U - L) * quantitle / (n * 2 * np.sqrt(2))
        )

    # prepare return values
    mu_up = np.zeros(N)
    mu_low = np.zeros(N)
    Sn_up = np.zeros(N)
    Sn_low = np.zeros(N)

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    sq_diff = 0.0
    Ahat = np.inf
    Ymean = np.cumsum(Y) / np.arange(1, N + 1)
    Abar_min_prev = std_upper_bound

    kmax = np.ceil(math.log(N, eta))
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_1_k = delta_1 / h(k)
            delta_2_k = delta_2 / h(k)

            for n in np.arange(nmin, min(nmax, N) + 1):

                if n == 1:
                    Abar = std_upper_bound
                    g = 0.0
                else:
                    if n % 2 == 0:
                        sq_diff += (Y[n - 1] - Y[n - 2]) ** 2 / 2.0
                        Ahat = np.sqrt(sq_diff / np.floor(n / 2.0))
                    g = g_function(n, nmax, delta_2_k)
                    Abar = np.sqrt(Ahat ** 2 + g ** 2) + g

                Abar_min = min(Abar, Abar_min_prev)

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_up[n - 2] != mu_up[n - 3]
                ):
                    q_up_func = QFunc(
                        n=nmax, A=Abar_min, B=mu_up[n - 2] - L if n >= 2 else U - L
                    )
                    Sn_low[n - 1] = -q_up_func(delta_1_k / 2.0)
                else:
                    Sn_low[n - 1] = Sn_low[n - 2]

                ### No minimum anymore
                mu_up[n - 1] = Ymean[n - 1] - Sn_low[n - 1] / n

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_low[n - 2] != mu_low[n - 3]
                ):
                    q_low_func = QFunc(
                        n=nmax, A=Abar_min, B=U - mu_low[n - 2] if n >= 2 else U - L
                    )
                    Sn_up[n - 1] = q_low_func(delta_1_k / 2.0)
                else:
                    Sn_up[n - 1] = Sn_up[n - 2]

                ### No maximum anymore
                mu_low[n - 1] = Ymean[n - 1] - Sn_up[n - 1] / n

                Abar_min_prev = Abar_min

    # print("final Abar_min: {}".format(Abar_min))
    return mu_low, mu_up, Sn_low, Sn_up


def adaptive_bentkus_stopping(Y, delta, U, L, eta=1.1, power=1.1, eps=0.1):
    """
        Adaptive Stopping algorithm for (eps, delta) mean estimation problem.
    """
    std_upper_bound = (U - L) / 2.0

    delta_1 = 2 * delta / 3  # for q function
    delta_2 = delta - delta_1  # for the upper bound of std

    N = Y.size

    c = scipy.special.zeta(x=power, q=1)

    def h(k):
        return (k + 1) ** power * c

    def g_function(n, nmax, delta_2_k):
        quantitle = scipy.stats.norm.ppf(1 - 2 * delta_2_k / np.exp(2))
        return (
            np.sqrt(np.floor(nmax / 2.0)) * (U - L) * quantitle / (n * 2 * np.sqrt(2))
        )

    # prepare return values
    mu_up = np.zeros(N)
    mu_low = np.zeros(N)
    Sn_up = np.zeros(N)
    Sn_low = np.zeros(N)

    # iterate through all intervals [ceil(eta**k), floor(eta**(k+1))]
    Abar_min_prev = std_upper_bound
    sq_diff = 0.0
    Ahat = np.inf
    Ymean = np.cumsum(Y) / np.arange(1, N + 1)

    LB = 0
    UB = np.inf
    found = False

    kmax = np.ceil(math.log(N, eta))
    for k in np.arange(max(kmax, 1)):
        nmin = int(np.ceil((eta ** k)))
        nmax = int(np.floor(eta ** (k + 1)))
        if nmax >= nmin:
            delta_1_k = delta_1 / h(k)
            delta_2_k = delta_2 / h(k)
            for n in np.arange(nmin, min(nmax, N) + 1):

                if n == 1:
                    Abar = std_upper_bound
                    g = 0.0
                else:
                    if n % 2 == 0:
                        sq_diff += (Y[n - 1] - Y[n - 2]) ** 2 / 2.0
                        Ahat = np.sqrt(sq_diff / np.floor(n / 2.0))
                    g = g_function(n, nmax, delta_2_k)
                    Abar = np.sqrt(Ahat ** 2 + g ** 2) + g

                Abar_min = min(Abar, Abar_min_prev)

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_up[n - 2] != mu_up[n - 3]
                ):
                    q_up_func = QFunc(
                        n=nmax, A=Abar_min, B=mu_up[n - 2] - L if n >= 2 else U - L
                    )
                    Sn_low[n - 1] = -q_up_func(delta_1_k / 2.0)
                else:
                    Sn_low[n - 1] = Sn_low[n - 2]
                mu_up[n - 1] = min(
                    Ymean[n - 1] - Sn_low[n - 1] / n, mu_up[n - 2] if n >= 2 else U
                )

                if (
                    n <= 2
                    or n == nmin
                    or Abar_min != Abar_min_prev
                    or mu_low[n - 2] != mu_low[n - 3]
                ):
                    q_low_func = QFunc(
                        n=nmax, A=Abar_min, B=U - mu_low[n - 2] if n >= 2 else U - L
                    )
                    Sn_up[n - 1] = q_low_func(delta_1_k / 2.0)
                else:
                    Sn_up[n - 1] = Sn_up[n - 2]
                mu_low[n - 1] = max(
                    Ymean[n - 1] - Sn_up[n - 1] / n, mu_low[n - 2] if n >= 2 else L
                )

                Abar_min_prev = Abar_min

                Q = np.maximum(np.abs(Sn_low[n - 1]), np.abs(Sn_up[n - 1])) / n
                LB = max(LB, np.abs(Ymean[n - 1]) - Q)
                UB = min(UB, np.abs(Ymean[n - 1]) + Q)
                if (1 + eps) * LB >= (1 - eps) * UB:
                    found = True
                    break

            if found:
                break
    mu_hat = 0.5 * np.sign(Ymean[n - 1]) * ((1 + eps) * LB + (1 - eps) * UB)
    if found:
        print("Stop:{}  Mu_hat:{}".format(n, mu_hat))
    else:
        print("Didn't found.")
    return n, mu_hat
