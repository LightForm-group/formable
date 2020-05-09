"""`formable.load_response.py`"""

import copy
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.optimize import least_squares

from formable.maths_utils import get_principal_values
from formable.utils import requires, at_most_one_of
from formable.yielding import (
    YieldFunction, YIELD_FUNCTION_MAP, DEF_3D_RES, DEF_2D_RES)
from formable.yielding.yield_point import (
    YieldPointUnsatisfiedError, init_yield_point_criteria)


class InhomogeneousDataError(Exception):
    pass


class LoadResponse(object):
    """A general class for representing the (simulated or real) response of applying a
    load to a (model) material."""

    ALLOWED_DATA = [
        'true_stress',
        'equivalent_strain',
    ]

    def __init__(self, **incremental_data):
        """
        Parameters
        ----------
        incremental_data : dict of (str : ndarray of shape (N, ...))
            A dict mapping the names of incremental quantities and their values as numpy
            arrays. Different combinations of quantities are required for different
            analysis. Arrays for all incremental data must have the same outer shape (N,),
            which is the number of increments over which the response is considered.

        Notes
        -----
        In future, we can add some more strain scalars that might required additional
        processing; e.g. the deformation gradient tensor, F, could be passed. It would
        need to be changed to a strain tensor, and then to a scalar. In this
        case the `yield_point_value` could be a dict, allowing additional parameters to
        be passed (the decorator wouldn't need to change). Also, we might want to define
        the yield point condition more generally, e.g. using a tensor.

        """

        if not incremental_data:
            msg = ('Pass at least one incremental data array. Allowed data is {}.')
            raise ValueError(msg.format(self.allowed_data_fmt))

        self._true_stress = incremental_data.pop('true_stress', None)
        self._equivalent_strain = incremental_data.pop('equivalent_strain', None)

        if incremental_data:
            unknown_fmt = ', '.join(['"{}"'.format(i) for i in incremental_data])
            msg = ('Unknown incremental data "{}". Allowed incremental data are: {}')
            raise ValueError(msg.format(unknown_fmt, self.allowed_data_fmt))

    @property
    def incremental_data_names(self):
        'Which of the allowed incremental data does this LoadResponse have?'
        has = []
        for allowed in self.ALLOWED_DATA:
            if getattr(self, allowed) is not None:
                has.append(allowed)
        return has

    @staticmethod
    def required_incremental_data(cls, analysis_type=None):
        """Get a list of the required incremental data for a given type of analysis."""
        pass

    @property
    def allowed_data_fmt(self):
        'Get a comma-separated list of allowed incremental data.'
        return ', '.join(['"{}"'.format(i) for i in self.ALLOWED_DATA])

    @property
    def true_stress(self):
        return self._true_stress

    @property
    def equivalent_strain(self):
        return self._equivalent_strain

    @property
    @requires('true_stress')
    def principal_true_stress(self):
        """Get principal true stress values."""
        return get_principal_values(self.true_stress)

    @requires('true_stress')
    @init_yield_point_criteria
    def get_yield_stress(self, yield_point_criteria, value_idx=None):
        """Find the yield stresses associated with a yield point criteria.

        Parmeters
        ---------
        yield_point_criteria : YieldPointCriterion or dict
            If a dict, it must be a dict of keyword arguments that can be used to
            instantiate a YieldPointCriterion object.

        Returns
        -------
        Two-tuple of:
            yield_stress : ndarray of shape (M, 3, 3)
                Yield stress tensors, one for each of M yield point criteria values.
            good_idx : list of int
                Indices of yield point criteria values for which the yield stress was
                successfully calculated.

        """

        source_dat = getattr(self, yield_point_criteria.source)
        yield_stress, good_idx = yield_point_criteria.get_yield_stress(
            source_dat, self.true_stress, value_idx=value_idx)

        return yield_stress, good_idx

    @init_yield_point_criteria
    def get_principal_yield_stress(self, yield_point_criteria, value_idx=None):
        """Find the principal yield stresses associated with a yield point criteria.

        Parmeters
        ---------
        yield_point_criteria : YieldPointCriterion or dict
            If a dict, it must be a dict of keyword arguments that can be used to
            instantiate a YieldPointCriterion object.

        Returns
        -------
        yld_stress_principal : ndarray of shape (M, 3)
            Principal yield stress tensors, one for each of M yield point criteria values,
            ordered from largest to smallest.
        value_idx : list of int
            Indices of values for which the yield stress was successfully calculated.

        """

        yld_stress, good_idx = self.get_yield_stress(yield_point_criteria, value_idx)

        if yld_stress.shape:
            yld_stress_principle = get_principal_values(yld_stress)
        else:
            yld_stress_principle = yld_stress

        return yld_stress_principle, good_idx

    @requires('true_stress')
    def is_uniaxial(self, increment=-1, tol=1e-3):
        """Is the specified increment's true stress state approximately uniaxial?"""

        princ_stress = self.principal_true_stress[increment]

        # Principal values are ordered largest to smallest, so check the first is much
        # larger than the other two:
        normed = princ_stress / princ_stress[0]
        if (abs(normed[1]) - tol) <= 0 and (abs(normed[2]) - tol) <= 0:
            return True
        else:
            return False


class LoadResponseSet(object):

    def __init__(self, load_responses):
        """

        Parameters
        ----------
        load_responses : list of LoadResponse
            All LoadResponse objects within the list must have the same set of specified
            incremental data.

        """

        # Validate:
        inc_data = load_responses[0].incremental_data_names
        for i in load_responses[1:]:
            if i.incremental_data_names != inc_data:
                msg = ('All load responses within a {} must have the same set of '
                       'specified incremental data.')
                raise InhomogeneousDataError(msg.format(self.__class__.__name__))

        self.responses = load_responses

        self.yield_point_criteria = []  # Appended in `self.calculate_yield_stresses`
        self.yield_stresses = []        # Appended in `self.calculate_yield_stresses`
        self.yield_functions = []       # Appended in `self.calculate_yield_function_fit`

    def __len__(self):
        return len(self.responses)

    @init_yield_point_criteria
    def calculate_yield_stresses(self, yield_point_criteria):
        """Calculate and store yield stresses for each load case and the specified
        yield point criteria.

        Parameters
        ----------
        yield_point_criteria : YieldPointCriterion or dict
            If a dict, it must be a dict of keyword arguments that can be used to
            instantiate a YieldPointCriterion object.

        """

        for ypv_idx, yield_point_val in enumerate(yield_point_criteria.values):

            yield_stress_dict = {
                'YPC_idx': len(self.yield_point_criteria),
                'YPC_value_idx': ypv_idx,
                'values': [],
                'response_idx': [],
            }

            for resp_idx, resp_i in enumerate(self.responses):

                val, _ = resp_i.get_yield_stress(yield_point_criteria, value_idx=ypv_idx)

                if val.size:
                    yield_stress_dict['values'].append(val)
                    yield_stress_dict['response_idx'].append(resp_idx)

            yield_stress_dict['values'] = np.array(yield_stress_dict['values'])

            if not len(yield_stress_dict['values']):
                msg = (f'No yield stresses were found for yield point criteria '
                       f'value {yield_point_val}.')
                raise ValueError(msg)

            self.yield_stresses.append(yield_stress_dict)

        self.yield_point_criteria.append(yield_point_criteria)

    def _validate_yield_function_parameters(self, yield_function,
                                            yield_point_criteria_idx,
                                            yield_point_criteria_value_idx,
                                            uniaxial_response, **kwargs):

        ypc = self.yield_point_criteria[yield_point_criteria_idx]

        uniaxial_eq_stress = None
        if 'equivalent_stress' in yield_function.PARAMETERS:

            if 'equivalent_stress' not in kwargs:

                if uniaxial_response is not None:
                    # equivalent stress is calculated as the yield stress in the
                    # uniaxial response, for a given specified yield point:

                    if not isinstance(uniaxial_response, LoadResponse):
                        raise TypeError('`uniaxial_response` must be a LoadResponse')

                    if not uniaxial_response.is_uniaxial(tol=5e-3):
                        msg = ('Specified `uniaxial_response` does not appear to be '
                               'uniaxial.')
                        raise ValueError(msg)

                    princ_stress, _ = uniaxial_response.get_principal_yield_stress(
                        ypc, value_idx=yield_point_criteria_value_idx)

                    if not princ_stress.size:
                        msg = ('Yield point not reached within uniaxial response.')
                        raise ValueError(msg)

                    # Turn into scalars:
                    uniaxial_eq_stress = princ_stress[0]

        else:

            msg = (f'The yield function {yield_function.__name__} does not require '
                   'an equivalent stress parameter, so `{}` is not required.')
            if uniaxial_response is not None:
                raise ValueError(msg.format('uniaxial_response'))
            if kwargs.get('equivalent_stress') is not None:
                raise ValueError(msg.format('equivalent_stress'))

        kwargs_copy = copy.deepcopy(kwargs)
        if uniaxial_eq_stress:
            kwargs_copy.update({
                'equivalent_stress': uniaxial_eq_stress,
            })

        return kwargs_copy

    @at_most_one_of('equivalent_stress', 'uniaxial_response')
    def fit_yield_function(self, yield_function, yield_point_criteria_idx=None,
                           uniaxial_response=None, initial_params=None, **kwargs):
        """Perform a fit to a yield function of all computed yield stresses.

        Parameters
        ----------
        yield_function : str or YieldFunction class
            The yield function to fit. Available yield functions can be displayed using: 
                `from formable import AVAILABLE_YIELD_FUNCTIONS`
                `print(AVAILABLE_YIELD_FUNCTIONS)`
        yield_point_criteria_idx : list of (list of int of length 2), optional
            If specified, fit the yield function to only the yield stresses calculated
            from the specified yield point criteria value. By default, None, in which case
            the yield function is fitted to all available yield stresses.
        initial_params : dict, optional
            Any initial guesses for the fitting parameters. Mutually exclusive with
            additional keyword arguments passed, which are considered to be fixed.            
        kwargs : dict
            Any yield function parameters to be fixed during the fit can be passed as
            additional keyword arguments.

        """

        # Validation:
        msg = 'Yield function "{}" not known. Available yield functions are: {}'
        available = ', '.join(['"{}"'.format(i) for i in YIELD_FUNCTION_MAP])
        msg = msg.format(yield_function, available)

        if isinstance(yield_function, str):
            if yield_function not in YIELD_FUNCTION_MAP:
                raise ValueError(msg)
            yield_function = YIELD_FUNCTION_MAP[yield_function]

        elif not issubclass(yield_function, YieldFunction):
            raise TypeError(msg)

        bad_kwargs = list(set(kwargs.keys()) - set(yield_function.PARAMETERS))
        if bad_kwargs:
            bad_kwargs_fmt = ', '.join([f'"{i}"' for i in bad_kwargs])
            raise ValueError(f'Unknown yield function parameters: {bad_kwargs_fmt}')

        if not yield_point_criteria_idx:
            yield_point_criteria_idx = [
                [ypc_idx, ypc_val_idx]
                for ypc_idx, ypc in enumerate(self.yield_point_criteria)
                for ypc_val_idx in range(len(ypc))
            ]

        for ypc_idx, ypc_val_idx in yield_point_criteria_idx:

            yield_stress = None
            yield_stress_idx = None
            for ys_idx, ys in enumerate(self.yield_stresses):
                if ys['YPC_idx'] == ypc_idx and ys['YPC_value_idx'] == ypc_val_idx:
                    yield_stress = ys['values']
                    yield_stress_idx = ys_idx
                    break

            if yield_stress is None:
                msg = (f'No yield stress found corresponding to '
                       f'`yield_point_criteria_idx={yield_point_criteria_idx}`.')
                raise ValueError(msg)

            yield_func_dict = {
                'YPC_idx': ypc_idx,
                'YPC_value_idx': ypc_val_idx,
                'yield_stress_idx': yield_stress_idx,
                'yield_function': None,
            }
            updated_kwargs = self._validate_yield_function_parameters(
                yield_function,
                ypc_idx,
                ypc_val_idx,
                uniaxial_response,
                **kwargs
            )

            # Perform fit:
            yld_func_obj = yield_function.from_fit(
                yield_stress,
                initial_params=initial_params,
                **updated_kwargs
            )

            ypc = self.yield_point_criteria[ypc_idx]
            yld_func_obj.yield_point = ypc.get_formatted(values_idx=ypc_val_idx)
            yield_func_dict['yield_function'] = yld_func_obj

            self.yield_functions.append(yield_func_dict)

    def remove_yield_function_fits(self):
        'Remove all yield function fits'

        self.yield_functions = []

    def show_yield_functions_3D(self, normalise=True, resolution=DEF_3D_RES,
                                equivalent_stress=None, min_stress=None, max_stress=None,
                                show_axes=True, planes=None, backend='plotly',
                                join_stress_states=False, show_contour_grid=False):
        'Visualise all fitted yield functions and data in 3D.'

        if not self.yield_functions:
            raise ValueError('No yield functions have been fitted to the load set.')

        yld_funcs = []
        yld_stresses = []
        stress_indices = []
        for yld_func_dict in self.yield_functions:
            yld_funcs.append(yld_func_dict['yield_function'])
            yld_stress = self.yield_stresses[yld_func_dict['yield_stress_idx']]
            yld_stress_vals = yld_stress['values']
            resp_idx = yld_stress['response_idx']
            yld_stress_principal = get_principal_values(yld_stress_vals)
            yld_stresses.append(yld_stress_principal)
            stress_indices.append(resp_idx)

        return YieldFunction.compare_3D(
            yld_funcs,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            show_axes=show_axes,
            planes=planes,
            stress_states=yld_stresses,
            stress_indices=stress_indices,
            backend=backend,
            join_stress_states=join_stress_states,
            show_contour_grid=show_contour_grid,
        )

    def show_yield_functions_2D(self, plane, normalise=True, resolution=DEF_2D_RES,
                                equivalent_stress=None, min_stress=None, max_stress=None,
                                show_axes=True, up=None, show_contour_grid=False,
                                join_stress_states=False):
        'Visualise all fitted yield functions and data in 2D.'

        if not self.yield_functions:
            raise ValueError('No yield functions have been fitted to the load set.')

        yld_funcs = []
        yld_stresses = []
        stress_indices = []
        for yld_func_dict in self.yield_functions:
            yld_funcs.append(yld_func_dict['yield_function'])
            yld_stress = self.yield_stresses[yld_func_dict['yield_stress_idx']]
            yld_stress_vals = yld_stress['values']
            resp_idx = yld_stress['response_idx']
            yld_stress_principal = get_principal_values(yld_stress_vals)
            yld_stresses.append(yld_stress_principal)
            stress_indices.append(resp_idx)

        return YieldFunction.compare_2D(
            yld_funcs,
            plane,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            stress_states=yld_stresses,
            stress_indices=stress_indices,
            up=up,
            show_contour_grid=show_contour_grid,
            join_stress_states=join_stress_states,
        )
