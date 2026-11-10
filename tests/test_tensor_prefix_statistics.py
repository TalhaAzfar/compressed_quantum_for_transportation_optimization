from fractions import Fraction
from math import comb
import numpy as np
from scipy.stats import hypergeom
from revision.audit_tensor_prefix_statistics import conditional_two_sided, holm_adjust


def test_tiny_conditional_tail_is_not_limited_by_fft_roundoff():
    tables = [(85, 8192, 6, 8192), (95, 8192, 3, 8192)]
    # Independently sum the upper-tail joint distribution, rather than convolving.
    probability = sum(hypergeom.pmf(k, 16384, 91, 8192) *
                      hypergeom.sf(179-k, 16384, 98, 8192) for k in range(92))
    actual = conditional_two_sided(tables)
    assert actual < 1e-40
    assert np.isclose(actual, 2*probability, rtol=1e-12, atol=0.)


def test_conditional_distribution_matches_exact_rational_enumeration():
    tables = [(4, 5, 1, 6), (2, 4, 0, 5)]
    distribution = {0: Fraction(1)}
    for a, na, b, nb in tables:
        next_distribution = {}
        for previous, probability in distribution.items():
            for k in range(max(0, a+b-nb), min(na, a+b)+1):
                mass = Fraction(comb(na,k)*comb(nb,a+b-k), comb(na+nb,a+b))
                next_distribution[previous+k] = next_distribution.get(previous+k, Fraction(0)) + probability*mass
        distribution = next_distribution
    observed = sum(t[0] for t in tables)
    expected = min(Fraction(1), 2*min(sum(p for k,p in distribution.items() if k <= observed),
                                     sum(p for k,p in distribution.items() if k >= observed)))
    assert np.isclose(conditional_two_sided(tables), float(expected), rtol=1e-13, atol=0.)


def test_zero_events_and_holm_family():
    assert conditional_two_sided([(0, 8192, 0, 8192)]*2) == 1.
    assert np.allclose(holm_adjust([.04, .001, .03, 1.]), [.09, .004, .09, 1.])
