import numpy as np
from scipy.stats import fisher_exact

from revision.analyze_nighthawk import difference_interval, holm, stratified_exact, wilson


def test_holm_known_adjustment():
    assert np.allclose(holm([.01,.04,.03,.002]), [.03,.06,.06,.008])


def test_zero_event_is_not_zero_probability_certainty():
    low, high = wilson(0, 100)
    assert low == 0
    assert 0 < high < .05
    low, high = difference_interval(0,100,0,100)
    assert low < 0 < high


def test_stratified_single_block_matches_symmetric_fisher():
    for a,b in [(0,0),(0,4),(1,7),(2,4),(10,1)]:
        expected = fisher_exact([[a,20-a],[b,20-b]]).pvalue
        assert np.isclose(stratified_exact([(a,20,b,20)]), expected, atol=1e-12)


def test_block_stratified_test_is_invariant_to_arm_swap():
    tables = [(2,40,7,40),(9,40,12,40),(0,40,1,40)]
    swapped = [(b,nb,a,na) for a,na,b,nb in tables]
    assert np.isclose(stratified_exact(tables), stratified_exact(swapped))
