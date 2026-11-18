"""Tests for the Levenberg-Marquardt fitting helpers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from formable.levenberg_marquardt import (
    FittingParameter,
    LMFitter,
    LMFitterOptimisation,
)


class FittingParameterTestCase(unittest.TestCase):

    def test_bounds_limit_perturbations_in_physical_units(self):
        param = FittingParameter(
            'strength', [10], ['material'], 0.1, bounds=(5, 12)
        )

        self.assertAlmostEqual(param.get_perturbed_value(0) * param.scale, 11)

        param.values[0] = 1.2
        self.assertAlmostEqual(param.get_perturbation(0), -0.12)

    def test_initial_value_must_be_within_bounds(self):
        with self.assertRaises(ValueError):
            FittingParameter('strength', [10], [], 0.1, bounds=(11, 20))

    def test_bounds_and_scale_survive_serialization(self):
        param = FittingParameter(
            'strength', [10], ['material'], 0.1, bounds=(5, 12)
        )
        restored = FittingParameter(**param.to_dict())

        self.assertEqual(restored.initial_value, 10)
        self.assertEqual(restored.bounds, (5, 12))


class LMFitterOptimisationTestCase(unittest.TestCase):

    def _optimisation_with_delta(self, residual, bounds=None):
        param = FittingParameter(
            'strength', [10], [], 0.1, bounds=bounds
        )
        fitter = SimpleNamespace(
            fitting_params=[param],
            optimisations=[],
            initial_damping=[1],
            num_params=1,
        )
        opt = object.__new__(LMFitterOptimisation)
        opt.lm_fitter = fitter
        opt._jacobian = np.array([[1.0]])
        opt.get_exp_stress = lambda: np.array([residual])
        opt.get_sim_stress = lambda sim_idx: np.array([0.0])
        return opt

    def test_step_limit_caps_both_positive_and_negative_updates(self):
        positive_opt = self._optimisation_with_delta(100)
        negative_opt = self._optimisation_with_delta(-100)

        self.assertAlmostEqual(positive_opt.find_new_parameters()[0], 1.5)
        self.assertAlmostEqual(negative_opt.find_new_parameters()[0], 0.5)

    def test_parameter_bounds_limit_proposed_update(self):
        opt = self._optimisation_with_delta(-100, bounds=(8, 20))

        self.assertAlmostEqual(opt.find_new_parameters()[0] * 10, 8)

    def test_worsened_actual_error_increases_next_damping_trials(self):
        fitter = SimpleNamespace(optimisations=[
            SimpleNamespace(damping_factor=1, actual_error=4),
            SimpleNamespace(damping_factor=1, actual_error=5),
        ])
        opt = object.__new__(LMFitterOptimisation)
        opt.lm_fitter = fitter

        self.assertEqual(opt.trial_damping_factors, [10, 4, 2])

    def test_worse_simulation_rejects_candidate_and_reuses_accepted_jacobian(self):
        param = FittingParameter('strength', [10], [], 0.1)
        param.values = np.append(param.values, 0.9)
        previous_opt = SimpleNamespace(
            actual_error=5,
            damping_factor=1,
            find_new_parameters=lambda damping: [0.8],
        )
        fitter = object.__new__(LMFitter)
        fitter.optimisations = [previous_opt]
        fitter.fitting_params = [param]

        class SimulatedOpt:
            def __init__(self, lm_fitter, tensile_tests):
                self._actual_error = None
                pass

            @property
            def actual_error(self):
                return self._actual_error

            def get_exp_stress(self):
                return np.array([10])

            def get_sim_stress(self, sim_idx):
                return np.array([0])

        with patch(
                'formable.levenberg_marquardt.LMFitterOptimisation', SimulatedOpt):
            accepted = fitter.add_simulated_tensile_tests([])

        self.assertFalse(accepted)
        self.assertEqual(param.values[-1], 0.8)
        self.assertEqual(fitter.optimisations, [previous_opt])


if __name__ == '__main__':
    unittest.main()
