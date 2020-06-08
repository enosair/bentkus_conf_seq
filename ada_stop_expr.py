import pickle
import time

import numpy as np

from conc_ineq.bentkus import adaptive_bentkus_stopping
from conc_ineq.bernstein import (empirical_bernstein_seq,
                                 hrms_bernstein_gamma_seq, hrms_bernstein_seq)
from conc_ineq.hoeffding import adaptive_hoeffding_seq


def gen_data(m, N, L=0, U=1):
    Y = np.random.uniform(L, U, size=(m, N))
    Y = np.sum(Y, axis=0) / m
    return Y


def ada_stop(Ymean, Q, eps=0.1):
    N = Q.size
    LB = 0
    UB = np.inf

    found = False
    for n in np.arange(N):

        LB = max(LB, np.abs(Ymean[n]) - Q[n])
        UB = min(UB, np.abs(Ymean[n]) + Q[n])
        if (1 + eps) * LB >= (1 - eps) * UB:
            found = True
            break

    stop_time = n + 1
    mu_hat = 0.5 * np.sign(Ymean[n]) * ((1 + eps) * LB + (1 - eps) * UB)

    if not found:
        print("Didn't find an accurate solution.")
    else:
        print("Stop:{}  Mu_hat:{}".format(stop_time, mu_hat))
    return stop_time, mu_hat


def run_ada_stop_expr(m, nrep, seed, eps=0.1):
    N = 5000
    eta = 1.1
    power = 1.1
    delta = 0.05
    start = time.time()
    nrep = nrep

    np.random.seed(seed)

    stop_ah = np.zeros(nrep)
    mu_ah = np.zeros(nrep)
    stop_bn = np.zeros(nrep)
    mu_bn = np.zeros(nrep)
    stop_ho = np.zeros(nrep)
    mu_ho = np.zeros(nrep)
    stop_bk = np.zeros(nrep)
    mu_bk = np.zeros(nrep)
    stop_gamma = np.zeros(nrep)
    mu_gamma = np.zeros(nrep)

    for ii in range(nrep):
        print("======= Rep {} =======".format(ii))
        Y = gen_data(m=m, N=N, L=0, U=1)
        ns = np.arange(1, N + 1)
        Ymean = np.cumsum(Y) / ns

        print("AH:")
        f_ah = adaptive_hoeffding_seq(N, delta / 2, upper=1, lower=0)
        stop_ah[ii], mu_ah[ii] = ada_stop(Ymean, Q=f_ah / ns, eps=eps)

        print("\n\nBN:")
        f_bn = empirical_bernstein_seq(
            Y, delta / 2, B=1.0, eta=eta, power=power, std_upper_bound=0.5
        )
        stop_bn[ii], mu_bn[ii] = ada_stop(Ymean, Q=f_bn / ns, eps=eps)

        print("\n\nHO:")
        f_ho = hrms_bernstein_seq(Y, delta / 2, std_upper_bound=0.5)
        stop_ho[ii], mu_ho[ii] = ada_stop(Ymean, Q=f_ho / ns, eps=eps)

        print("\n\nHRMS-Gamma:")
        f_gamma = hrms_bernstein_gamma_seq(Y, delta / 2, L=0.0, U=1.0, v_opt=10)
        stop_gamma[ii], mu_gamma[ii] = ada_stop(Ymean, Q=f_gamma / ns, eps=eps)

        print("\n\nBK:")
        stop_bk[ii], mu_bk[ii] = adaptive_bentkus_stopping(
            Y, delta, U=1.0, L=0.0, eta=eta, power=power, eps=eps
        )

    runtime = time.time() - start
    print("Runtime {}".format(runtime))


#    with open('./expr/stop_m{}_seed{}.pickle'.format(m, seed), 'wb') as f:
#        rst = {'stop_ah': stop_ah,
#               'stop_bn': stop_bn,
#               'stop_bk': stop_bk,
#               'stop_ho': stop_ho,
#               'stop_gamma': stop_gamma,
#               'mu_ah': mu_ah,
#               'mu_bn': mu_bn,
#               'mu_ho': mu_ho,
#               'mu_bk': mu_bk,
#               'mu_gamma': mu_gamma,
#              }
#        pickle.dump(rst, f)
run_ada_stop_expr(m=100, nrep=3, seed=123, eps=0.1)
