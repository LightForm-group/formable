'Module containing unit tests on `formable.yielding.yield_point.py` functionality.'

import unittest

import numpy as np

from formable.yielding.yield_point import (
    YieldPointUnsatisfiedError,
    get_yield_stress_from_equivalent_strain,
)


class YieldStressFromEquivalentStrainTestCase(unittest.TestCase):
    """Tests on `get_yield_stress_from_equivalent_strain`."""

    def test_expected_yield_stress(self):
        'Test expected pre-determined yield stress is returned.'

        # From 0.000 to 0.001 in steps of 0.0001:
        num_increments = 11
        equiv_strain = np.linspace(0, 1e-3, num=num_increments)
        stress = np.random.random((num_increments, 3, 3))

        # This yield point should be halfway between `equiv_strain` indices 3 and 4:
        yld_point = 0.00035
        yld_stress_exp = (stress[3] + stress[4]) / 2

        yld_stress = get_yield_stress_from_equivalent_strain(
            equiv_strain, yld_point, stress)

        self.assertTrue(np.allclose(yld_stress, yld_stress_exp))

    def test_raise_on_unsatisfied_yield_point(self):
        'Test raise on unachieved yield point.'

        # From 0.000 to 0.001 in steps of 0.0001:
        num_increments = 11
        equiv_strain = np.linspace(0, 1e-3, num=num_increments)
        stress = np.random.random((num_increments, 3, 3))

        # This yield point is beyond the values within `equiv_strain`:
        yld_point = 0.002

        with self.assertRaises(YieldPointUnsatisfiedError):
            get_yield_stress_from_equivalent_strain(equiv_strain, yld_point, stress)

    def test_raise_on_insufficient_data(self):
        'Test raise when the yield point is reached after the first increment.'

        # From 0.0002 to 0.001 in steps of 0.0001:
        num_increments = 9
        equiv_strain = np.linspace(2e-4, 10e-4, num=num_increments)
        stress = np.random.random((num_increments, 3, 3))

        # This yield point is lower than the first value increment of `equiv_strain`,
        # meaning two increment indices from which to interpolate the stress do not exist:
        yld_point = 0.0001

        with self.assertRaises(YieldPointUnsatisfiedError):
            get_yield_stress_from_equivalent_strain(equiv_strain, yld_point, stress)
