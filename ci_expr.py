import argparse
import os
import pickle
import time

import numpy as np

from conc_ineq.bentkus import adaptive_bentkus_seq
from conc_ineq.bernstein import (empirical_bernstein_seq,
                                 hrms_bernstein_gamma_seq, hrms_bernstein_seq)
from conc_ineq.hoeffding import adaptive_hoeffding_seq


def compute_failure(Ymean, ns, ff, mu):
    up = np.minimum(Ymean + ff / ns, 1.0)
    low = np.maximum(Ymean - ff / ns, 0.0)
    return np.any(mu <= low) or np.any(mu >= up)


def ci_expr(N, p, seed, delta=0.05, eta=1.1, power=1.1):

    start = time.time()

    # generate data: Yi ~ Bernoulli(p)
    np.random.seed(seed)
    Y = np.random.binomial(1, p=p, size=(N,))

    # Compute bounds
    f_ah = adaptive_hoeffding_seq(N, delta / 2, upper=1, lower=0)

    f_bn = empirical_bernstein_seq(
        Y, delta / 2, B=1.0, eta=eta, power=power, std_upper_bound=0.5
    )

    f_ho = hrms_bernstein_seq(Y, delta / 2, std_upper_bound=0.5)
    f_gamma = hrms_bernstein_gamma_seq(Y, delta / 2, L=0.0, U=1.0)

    mu_low, mu_up, _, _ = adaptive_bentkus_seq(
        Y, delta, U=1.0, L=0.0, eta=eta, power=power
    )

    # Compute failure probability
    ns = np.arange(1, N + 1)
    Sn = np.cumsum(Y)
    Ymean = Sn / ns
    mu = p

    fail_ah = compute_failure(Ymean, ns, f_ah, mu)
    fail_bn = compute_failure(Ymean, ns, f_bn, mu)
    fail_ho = compute_failure(Ymean, ns, f_ho, mu)
    fail_gamma = compute_failure(Ymean, ns, f_gamma, mu)
    fail_bk = np.any(mu <= mu_low) or np.any(mu >= mu_up)

    run_time = time.time() - start

    # Log and save
    print("Runtime: {}".format(run_time))
    print("fail_ah: {}".format(fail_ah))
    print("fail_bn: {}".format(fail_bn))
    print("fail_bk: {}".format(fail_bk))
    print("fail_ho: {}".format(fail_ho))
    print("fail_gamma: {}".format(fail_gamma))

    # if not os.path.exists("./expr/{}".format(N)):
    #     os.mkdir("./expr/{}".format(N))
    # with open("./expr/{}/ci_expr_{}.pickle".format(N, seed), "wb") as f:
    #     rst = {
    #         "Y": Y,
    #         "f_ah": f_ah,
    #         "f_bn": f_bn,
    #         "f_ho": f_ho,
    #         "f_gamma": f_gamma,
    #         "mu_low": mu_low,
    #         "mu_up": mu_up,
    #         "fail_ah": fail_ah,
    #         "fail_bn": fail_bn,
    #         "fail_bk": fail_bk,
    #         "fail_ho": fail_ho,
    #         "fail_gamma": fail_gamma,
    #     }
    #     pickle.dump(rst, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-NN", type=int, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-p", type=float, required=True)
    args = parser.parse_args()
    print("N = {} p = {} seed = {}".format(args.NN, args.p, args.seed))

    ci_expr(N=args.NN, p=args.p, seed=args.seed, delta=0.05, eta=1.1, power=1.1)


if __name__ == "__main__":
    main()
