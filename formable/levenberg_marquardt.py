'Fitting crystal plasticity parameters using a Levenberg-Marquardt optimisation process.'

import copy
import json
from pathlib import Path
from importlib import import_module

import numpy as np
from plotly import graph_objects

from formable.tensile_test import TensileTest
from formable.utils import find_nearest_index


class FittingParameter(object):
    'Represents a parameter to be fit and its fitted values.'

    def __init__(self, name, values, address, perturbation, scale=None):

        self.name = name
        self.address = address
        self.perturbation = perturbation

        self.values = np.array(values)
        self.scale = scale

        if self.values.size == 1:
            self.scale = values[0]
            self.values[0] = 1

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'name': self.name,
            'address': self.address,
            'perturbation': self.perturbation,
            'values': self.values.tolist(),
            'scale': self.scale,
        }
        return out

    def to_json_file(self, json_path):
        'Save as a JSON file'

        json_path = Path(json_path)
        dct = self.to_dict()
        with json_path.open('w') as handle:
            json.dump(dct, handle, sort_keys=True, indent=4)

        return json_path

    @classmethod
    def from_json_file(cls, json_path):
        'Load from a JSON file.'

        with Path(json_path).open() as handle:
            contents = json.load(handle)

        return cls(**contents)

    @property
    def initial_value(self):
        return self.values[0] * self.scale

    def get_perturbation(self, idx=-1):
        return self.values[idx] * self.perturbation

    def get_perturbed_value(self, idx=-1):
        return self.values[idx] + self.get_perturbation(idx)

    def __repr__(self):
        out = (
            '{}('
            'name={!r}, '
            'values={!r}'
            ')').format(
            self.__class__.__name__,
            self.name,
            self.values * self.scale,
        )
        return out


class LMFitterOptimisation(object):
    'Represents a single optimisation step in the LMFitter object.'

    def __init__(self, lm_fitter, sim_tensile_tests, _damping_factor=None, _error=None,
                 _delay_validation=False):

        # TODO: probably better to use __new__ somehow instead of this nonsense:
        if not _delay_validation:
            self.lm_fitter = lm_fitter
            self.sim_tensile_tests = self._validate_tensile_tests(sim_tensile_tests)

            exp_strain_idx, sim_strain_idx = self._get_strain_idx()

            self.exp_strain_idx = exp_strain_idx
            self.sim_strain_idx = sim_strain_idx
            self._jacobian = self._approximate_jacobian()
        else:
            self.sim_tensile_tests = sim_tensile_tests

        # These are assigned by `find_new_parameters`:
        self._damping_factor = _damping_factor
        self._error = _error

    def _validate(self, lm_fitter):

        self.lm_fitter = lm_fitter
        self.sim_tensile_tests = self._validate_tensile_tests(self.sim_tensile_tests)

        exp_strain_idx, sim_strain_idx = self._get_strain_idx()

        self.exp_strain_idx = exp_strain_idx
        self.sim_strain_idx = sim_strain_idx
        self._jacobian = self._approximate_jacobian()

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'sim_tensile_tests': [i.to_dict() for i in self.sim_tensile_tests],
            '_damping_factor': self.damping_factor,
            '_error': self.error,
        }
        return out

    def __repr__(self):
        out = (
            '{}('
            'error={}, '
            'damping_factor={}'
            ')'
        ).format(
            self.__class__.__name__,
            self.error,
            self.damping_factor,
        )
        return out

    @property
    def error(self):
        return self._error

    @property
    def damping_factor(self):
        return self._damping_factor

    @property
    def trial_damping_factors(self):
        if self.index > 0:
            prev_opt = self.lm_fitter.optimisations[self.index - 1]
            prev_damp = prev_opt.damping_factor
            out = [2 * prev_damp, prev_damp, 0.5 * prev_damp]
        else:
            out = self.lm_fitter.initial_damping

        return out

    @property
    def index(self):
        'Get the index of this optimisation step.'
        if self in self.lm_fitter.optimisations:
            # Already added to opt list:
            return self.lm_fitter.optimisations.index(self)
        else:
            # Not yet added to opt list:
            return len(self.lm_fitter.optimisations)

    def _validate_tensile_tests(self, sim_tensile_tests):
        'Check the correct number of tensile tests.'

        if len(sim_tensile_tests) != self.lm_fitter.sims_per_iteration:
            msg = ('There must be {} new tensile tests to add an optimisation '
                   'step, since there are {} fitting parameters.')
            raise ValueError(msg.format(self.lm_fitter.sims_per_iteration,
                                        self.lm_fitter.num_params))

        return sim_tensile_tests

    def get_exp_stress(self):
        return self.lm_fitter.exp_tensile_test.true_stress[self.exp_strain_idx]

    def get_exp_strain(self):
        return self.lm_fitter.exp_tensile_test.true_strain[self.exp_strain_idx]

    def get_sim_stress(self, sim_idx):
        return self.sim_tensile_tests[sim_idx].true_stress[self.sim_strain_idx[sim_idx]]

    def get_sim_strain(self, sim_idx):
        return self.sim_tensile_tests[sim_idx].true_strain[self.sim_strain_idx[sim_idx]]

    def _get_strain_idx(self):
        """Use the first simulated tensile test (unperturbed fitting parameters) to define
        the strain increments used in the Jacobian approximation.

        Returns
        -------
        tuple of (exp_strain_idx, sim_strain_idx)
            exp_strain_idx : ndarray of shape (N,)
            sim_strain_idx : ndarray of shape (M, N)

        """

        first_tt = self.sim_tensile_tests[0]
        exp_tt = self.lm_fitter.exp_tensile_test

        # Only use strain increments up to plastic upper limit (pre-necking) of the
        # experimental test (this assumes the simulated test goes beyond the plastic
        # limit):
        max_idx = find_nearest_index(first_tt.true_strain, exp_tt.plastic_range[1])

        exp_strain_idx = []
        sim_strain_idx = [[] for _ in range(len(self.sim_tensile_tests))]

        # Use all (up to max) strain increments in the first simulated response:
        sim_strain_idx[0] = np.arange(len(first_tt.true_strain[:max_idx]))

        for strain_val in first_tt.true_strain[:max_idx]:

            # Use experimental increments closest to the first simulated response:
            exp_strain_idx.append(find_nearest_index(exp_tt.true_strain, strain_val))

            # For each remaining sim, use increments closest to first simulated response:
            for j_idx, sim_tt in enumerate(self.sim_tensile_tests[1:], 1):
                sim_tt_idx = find_nearest_index(sim_tt.true_strain, strain_val)
                sim_strain_idx[j_idx].append(sim_tt_idx)

        exp_strain_idx = np.array(exp_strain_idx)
        sim_strain_idx = np.array(sim_strain_idx)

        return exp_strain_idx, sim_strain_idx

    def _approximate_jacobian(self):
        """Use the base and perturbed simulated tensile tests to approximate the Jacobian matrix.

        Notes
        -----
        Each row in the Jacobian matrix contains stress values from different strain
        increments. Each column in the Jacobian matrix represents the stress results
        from a simulation with a particular fitting parameter perturbed, minus the
        stress from the unperturbed simulation.

        """

        # Get stresses from perturbed sims:
        cols = [self.get_sim_stress(i) for i in range(1, self.lm_fitter.num_params + 1)]
        cols = np.vstack(cols).T

        # Subtract off stress from unperturbed sim:
        cols -= self.get_sim_stress(0)[:, None]

        perts = np.array([i.get_perturbation(self.index)
                          for i in self.lm_fitter.fitting_params])
        jacobian = cols / perts

        return jacobian

    @property
    def fitting_params(self):
        out = {}
        for param in self.lm_fitter.fitting_params:
            out.update({param.name: param.values[self.index] * param.scale})
        return out

    @property
    def jacobian(self):
        return self._jacobian

    def find_new_parameters(self):

        deltas = []
        errors = []

        jac_prod = self.jacobian.T @ self.jacobian
        right = self.jacobian.T @ (self.get_exp_stress() -
                                   self.get_sim_stress(0))[:, None]

        for damping in self.trial_damping_factors:

            left = jac_prod + damping * np.diag(jac_prod)
            delta = np.linalg.solve(left, right)
            stress_diff = (self.get_exp_stress() - self.get_sim_stress(0))[:, None]
            error = np.sum(np.abs(stress_diff - self.jacobian @ delta))
            deltas.append(delta)
            errors.append(error)

        best_idx = np.argmin(errors)
        best_delta = deltas[best_idx]

        self._damping_factor = self.trial_damping_factors[best_idx]
        self._error = errors[best_idx]

        cur_params = [i.values[-1] for i in self.lm_fitter.fitting_params]
        
        delta_ratio = [j/i for i, j in zip(cur_params, best_delta)]
        max_delta_ratio = max(delta_ratio)

        if max_delta_ratio > 0.5:
            delta_ratio = [i/max_delta_ratio*0.5 for i in delta_ratio]
        
        new_params = [i*(1+j) for i, j in zip(cur_params, delta_ratio)] 

        return new_params


class LMFitter(object):

    FIG_WIDTH = 480
    FIG_HEIGHT = 380
    FIG_MARG = {
        't': 80,
        'l': 60,
        'r': 50,
        'b': 80,
    }
    FIG_PAD = [0.01, 5]

    def __init__(self, exp_tensile_test, single_crystal_parameters, fitting_params,
                 initial_damping=None, optimisations=None):
        """Use a Levenberg-Marquardt optimisation process to find crystal plasticity
        hardening parameters.

        Parameters
        ----------
        exp_tensile_test : list of TensileTest
            The experimental data to which the parameters should be fit.            
        single_crystal_parameters : dict or list
            Dict or list of parameters that define the material properties in whatever
            simulation software is chosen. If a dict, it may be arbitrarily nested. Note:
            this should be JSON-compatible (i.e. no Numpy floats).
        fitting_params : list of FittingParameter
            List of FittingParameter objects, one for each parameter to be fit. The
            `address` attribute of each FittingParameter must point to a valid location
            within `single_crystal_parameters`.

        """

        self.exp_tensile_test = exp_tensile_test
        self.single_crystal_parameters = single_crystal_parameters
        self.optimisations = optimisations or []

        self.fitting_params = self._validate_fitting_params(fitting_params)
        self.initial_damping = initial_damping or [2, 1, 0.5]

        self._visual = None

    def _validate_fitting_params(self, fitting_params):
        'Check the correct numpy of fitting parameter values'
        for i in fitting_params:
            if len(i.values) != len(self.optimisations) + 1:
                msg = ('If there are N optimisations, each fitting parameter '
                       'must have N-1 values,')
                raise ValueError(msg)

        return fitting_params

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'exp_tensile_test': self.exp_tensile_test.to_dict(),
            'single_crystal_parameters': self.single_crystal_parameters,
            'fitting_params': [i.to_dict() for i in self.fitting_params],
            'optimisations': [i.to_dict() for i in self.optimisations],
            'initial_damping': self.initial_damping,
        }
        return out

    def to_json_file(self, json_path):
        """Save the state of the LMFitter object to a JSON file, to allow continuation of
        the fitting process at a later date."""

        json_path = Path(json_path)
        dct = self.to_dict()
        with json_path.open('w') as handle:
            json.dump(dct, handle, sort_keys=True, indent=4)

        return json_path

    @classmethod
    def from_dict(cls, lm_fitter_dict):
        'Load an LMFitter object from a JSON-compatbile dict.'

        # Don't modify the original:
        lm_fitter_dict = copy.deepcopy(lm_fitter_dict)

        lm_fitter_dict['exp_tensile_test'] = TensileTest(
            **lm_fitter_dict['exp_tensile_test']
        )

        lm_fitter_dict['fitting_params'] = [
            FittingParameter(**i)
            for i in lm_fitter_dict['fitting_params']
        ]

        for idx, i in enumerate(lm_fitter_dict['optimisations']):
            i.update({
                'lm_fitter': None,
                '_delay_validation': True,
                'sim_tensile_tests': [TensileTest(**j) for j in i['sim_tensile_tests']]
            })
            lm_fitter_dict['optimisations'][idx] = LMFitterOptimisation(**i)

        lm_fitter = cls(**lm_fitter_dict)
        for opt in lm_fitter.optimisations:
            opt._validate(lm_fitter)

        return lm_fitter

    @classmethod
    def from_json_file(cls, json_path):
        'Load an LMFitter from a JSON file.'

        with Path(json_path).open() as handle:
            contents = json.load(handle)

        lm_fitter = LMFitter.from_dict(contents)

        return lm_fitter

    @property
    def opt_index(self):
        return len(self.optimisations)

    @property
    def num_params(self):
        return len(self.fitting_params)

    @property
    def sims_per_iteration(self):
        return self.num_params + 1

    def get_parameter(self, name, address):
        'Get a parameter value from the `single_crystal_parameters` dict/list.'
        out = self.single_crystal_parameters
        for i in address:
            out = out[i]
        return out[name]

    @staticmethod
    def set_parameter(dct, name, address, value):
        print('set_parameter: setting ')
        for i in address:
            dct = dct[i]
        dct[name] = value

    def __repr__(self):
        out = ('{}('
               'fitting_params={!r}'
               ')').format(
            self.__class__.__name__,
            self.fitting_params,
        )
        return out

    def add_simulated_tensile_tests(self, tensile_tests):
        """Add simulation results to progress the optimisation process.

        Parameters
        ----------
        tensile_tests : list of TensileTest
            For M fitting parameters, this list must be of length M + 1. The first 
            TensileTest should be the zero-perturbation test. The remaining
            TensileTest objects should be given in the same order as the
            fitting parameters.

        """

        opt = LMFitterOptimisation(self, tensile_tests)
        self.optimisations.append(opt)
        new_params = opt.find_new_parameters()
        for i, j in zip(self.fitting_params, new_params):
            i.values = np.append(i.values, j)

    def get_new_single_crystal_params(self, fitting_idx):

        new_params = copy.deepcopy(self.single_crystal_parameters)
        for fit_param in self.fitting_params:

            LMFitter.set_parameter(
                new_params,
                fit_param.address[-1],
                fit_param.address[:-1],
                float(fit_param.values[fitting_idx]) * fit_param.scale,
            )
        return new_params

    def _generate_visual(self):

        data = [{
            'x': self.optimisations[0].get_exp_strain(),
            'y': self.optimisations[0].get_exp_stress() / 1e6,
            'mode': 'lines',
            'name': 'Exp.',
            'line': {
                'dash': 'dash',
            }
        }]
        data.extend([
            {
                'x': opt.get_sim_strain(0),
                'y': opt.get_sim_stress(0) / 1e6,
                'mode': 'lines',
                'name': 'Iter. {}'.format(idx),
                'visible': 'legendonly',
            } for idx, opt in enumerate(self.optimisations, 1)
        ])

        layout = {
            'title': 'Levenberg-Marquardt optimisation',
            'width': LMFitter.FIG_WIDTH,
            'height': LMFitter.FIG_HEIGHT,
            'margin': LMFitter.FIG_MARG,
            'xaxis': {
                'title': 'True strain, ε',
                'range': [
                    -LMFitter.FIG_PAD[0],
                    self.optimisations[0].get_exp_strain()[-1] + LMFitter.FIG_PAD[0]
                ],
            },
            'yaxis': {
                'title': 'True stress σ / MPa',
            }
        }

        fig = graph_objects.FigureWidget(data=data, layout=layout)

        return fig

    @property
    def visual(self):
        if not self._visual:
            self._visual = self._generate_visual()
        return self._visual

    def show(self, layout_args=None):
        viz = self.visual
        if layout_args:
            viz.layout.update(**layout_args)
        return viz
