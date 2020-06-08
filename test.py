import math
import unittest

import hypothesis.strategies as st
import numpy as np
import scipy
from hypothesis import given, settings

from conc_ineq.bentkus import adaptive_bentkus_seq_known_var, bentkus
from conc_ineq.bernstein import (empirical_bernstein,
                                 empirical_bernstein_seq_known_var)


class TestConcIneq(unittest.TestCase):
    @given(
        power=st.floats(min_value=1.01, max_value=1.9),
        eta=st.floats(min_value=1.01, max_value=1.9),
        A=st.floats(min_value=0.1, max_value=0.9),
        B=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=15, deadline=None)
    def test_bentkus_seq(self, power, eta, A, B):
        print("power={}, eta={}, A={}, B={}".format(power, eta, A, B))

        c = scipy.special.zeta(x=power, q=1)

        def h(k):
            return (k + 1) ** power * c

        fs = []
        N = 300
        delta = 0.05
        for n in range(1, N + 1):
            kn = int(math.log(n, eta))
            cn = int(eta ** (kn + 1))
            fs.append(
                bentkus(n=cn, delta=delta / h(kn), A=A, B=B)
            )
        f1 = adaptive_bentkus_seq_known_var(N, delta, A=A, B=B, eta=eta, power=power)
        self.assertFalse(np.any(f1 != fs))

    @given(
        power=st.floats(min_value=1.01, max_value=1.9),
        eta=st.floats(min_value=1.01, max_value=1.9),
        A=st.floats(min_value=0.1, max_value=0.9),
        B=st.floats(min_value=1.0, max_value=5.0),
    )
    @settings(max_examples=15, deadline=None)
    def test_empirical_bernstein(self, power, eta, A, B):
        print("power={}, eta={}, A={}, B={}".format(power, eta, A, B))

        fs = []
        N = 300
        delta = 0.03
        for n in range(1, N + 1):
            fs.append(
                empirical_bernstein(n=n, delta=delta, eta=eta, power=power, A=A, B=B)
            )
        f1 = empirical_bernstein_seq_known_var(N, delta, A=A, B=B, eta=eta, power=power)
        self.assertFalse(np.any(f1 != fs))


if __name__ == "__main__":
    unittest.main()
