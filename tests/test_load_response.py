'Module containing unit tests on `formable.load_response.py` functionality.'

import unittest

import numpy as np

from formable.load_response import LoadResponse, LoadResponseSet, InhomogeneousDataError
from formable.yielding import YieldPointCriteria


class LoadResponseTestCase(unittest.TestCase):
    'Tests on the LoadResponse class.'

    def test_is_uniaxial(self):
        'Test a known uniaxial load response reports as such.'

        num_increments = 10

        # Not a good proxy for realistic series of increasing stress states, but OK here:
        stress = np.random.random((num_increments, 3, 3))
        stress *= 1e4
        stress[:, 1, 1] *= 1e5  # The "uniaxial" component

        resp = LoadResponse(true_stress=stress)
        self.assertTrue(resp.is_uniaxial())

    def test_get_yield_stress_value_idx(self):
        """Check specifying `value_idx` in `LoadResponse.get_yield_stress` works as
        expected."""

        num_increments = 11
        equiv_strain = np.linspace(0, 1e-3, num=num_increments)
        stress = np.random.random((num_increments, 3, 3))

        resp = LoadResponse(true_stress=stress, equivalent_strain=equiv_strain)
        vals = [0.00035, 0.00045]
        yield_point_criteria = YieldPointCriteria('equivalent_strain', vals)

        yield_stress_all_vals = resp.get_yield_stress(yield_point_criteria)
        yield_stress_by_vals = [
            resp.get_yield_stress(yield_point_criteria, value_idx=i) for i in range(2)
        ]

        # Combine into dicts to compare (keyed by YPC value index):
        all_vals_dict = {ypc_val_idx: stress
                         for stress, ypc_val_idx in zip(*yield_stress_all_vals)}

        by_vals_dict = {ypc_val_idx[0]: stress
                        for stress, ypc_val_idx in yield_stress_by_vals}

        for ypc_val_idx, stress in all_vals_dict.items():
            self.assertTrue(np.allclose(stress, by_vals_dict[ypc_val_idx]))


class LoadResponseSetTestCase(unittest.TestCase):
    'Tests on the LoadResponseSet class.'

    def test_set_single_yield_stress_consistent(self):
        'Check yield stresses of set are consistent with those from individual responses.'

        num_increments = 11
        num_responses = 5
        equiv_strain = np.linspace(0, 1e-3, num=num_increments)
        stress = np.random.random((num_responses, num_increments, 3, 3))

        responses = [
            LoadResponse(true_stress=stress[idx], equivalent_strain=equiv_strain)
            for idx in range(num_responses)
        ]
        load_set = LoadResponseSet(responses)
        yield_point_criteria = YieldPointCriteria('equivalent_strain', 2e-4)
        load_set.calculate_yield_stresses(yield_point_criteria)

        ypc_idx = 0
        ypc_val_idx = 0
        resp_idx = 3

        for i in load_set.yield_stresses:
            if i['YPC_idx'] == ypc_idx and i['YPC_value_idx'] == ypc_val_idx:
                all_resp_idx = np.array(load_set.yield_stresses[ypc_idx]['response_idx'])
                value_idx = np.where(all_resp_idx == resp_idx)[0][0]
                yld_stress_all = load_set.yield_stresses[ypc_idx]['values'][value_idx]
                break

        yld_stress_resp = responses[resp_idx].get_yield_stress(
            yield_point_criteria,
            value_idx=ypc_val_idx,
        )[0]

        self.assertTrue(np.allclose(yld_stress_all, yld_stress_resp))

    def test_set_single_yield_stress_consistent_insufficient_yield(self):
        """Check yield stresses of set are consistent with those from individual
        responses, including responses with unachieved yield points."""

        num_increments = 11
        num_responses = 5
        equiv_strain = np.tile(
            np.linspace(0, 1e-3, num=num_increments),
            (num_responses, 1)
        )

        # Set the equivalent strain of one response to not meet the yield point:
        equiv_strain[1] /= 10
        stress = np.random.random((num_responses, num_increments, 3, 3))

        responses = [
            LoadResponse(true_stress=stress[idx], equivalent_strain=equiv_strain[idx])
            for idx in range(num_responses)
        ]
        load_set = LoadResponseSet(responses)
        yield_point_criteria = YieldPointCriteria('equivalent_strain', 2e-4)
        load_set.calculate_yield_stresses(yield_point_criteria)

        ypc_idx = 0
        ypc_val_idx = 0
        resp_idx = 3

        for i in load_set.yield_stresses:
            if i['YPC_idx'] == ypc_idx and i['YPC_value_idx'] == ypc_val_idx:
                all_resp_idx = np.array(load_set.yield_stresses[ypc_idx]['response_idx'])
                value_idx = np.where(all_resp_idx == resp_idx)[0][0]
                yld_stress_all = load_set.yield_stresses[ypc_idx]['values'][value_idx]
                break

        yld_stress_resp = responses[resp_idx].get_yield_stress(
            yield_point_criteria,
            value_idx=ypc_val_idx,
        )[0]

        self.assertTrue(np.allclose(yld_stress_all, yld_stress_resp))

    def test_raise_on_distinct_incremental_data(self):
        'Check LoadResponseSet raises when passed response objects with distinct data.'

        num_increments = 11
        equiv_strain = np.linspace(0, 1e-3, num=num_increments)
        stress = np.random.random((num_increments, 3, 3))

        response_1 = LoadResponse(true_stress=stress)
        response_2 = LoadResponse(equivalent_strain=equiv_strain)

        with self.assertRaises(InhomogeneousDataError):
            LoadResponseSet([response_1, response_2])

    def test_varying_response_length_OK(self):
        'Check LoadResponseSet can be initialised with response objects of varying length.'

        num_increments_1 = 11
        equiv_strain_1 = np.linspace(0, 1e-3, num=num_increments_1)
        stress_1 = np.random.random((num_increments_1, 3, 3))

        num_increments_2 = 5
        equiv_strain_2 = np.linspace(0, 1e-3, num=num_increments_2)
        stress_2 = np.random.random((num_increments_2, 3, 3))

        response_1 = LoadResponse(true_stress=stress_1, equivalent_strain=equiv_strain_1)
        response_2 = LoadResponse(true_stress=stress_2, equivalent_strain=equiv_strain_2)

        LoadResponseSet([response_1, response_2])
