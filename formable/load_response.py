"""`formable.load_response.py`"""

import copy
from collections import namedtuple
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.optimize import least_squares

from formable.maths_utils import get_principal_values
from formable.utils import requires, at_most_one_of
from formable.yielding import (
    YieldFunction,
    get_yield_function_map,
    DEF_3D_RES,
    DEF_2D_RES,
)
from formable.yielding.yield_point import (
    YieldPointUnsatisfiedError,
    init_yield_point_criteria,
    YieldPointCriteria
)


class InhomogeneousDataError(Exception):
    pass


class LoadResponse(object):
    """A general class for representing the (simulated or real) response of applying a
    load to a (model) material."""

    ALLOWED_DATA = [
        'stress',
        'equivalent_strain',
        'equivalent_plastic_strain',
        'accumulated_shear_strain',
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

        inc_data_valid_key = ''
        for k, v in incremental_data.items():
            if v is not None:
                inc_data_valid_key = k
                break
                
        outer_shape = incremental_data[inc_data_valid_key].shape[0]
        err = False
        for i in incremental_data.values():
            if i is None:
                continue
            try:
                if i.shape[0] != outer_shape:
                    err = True
            except AttributeError:
                err = True
            if err:
                msg = ('All `incremental_data` dict values must be Numpy arrays of '
                       'equal outer shape (corresponding to loading increments).')
                raise ValueError(msg)

        self._num_increments = outer_shape

        self._stress = incremental_data.pop('stress', None)
        self._equivalent_strain = incremental_data.pop('equivalent_strain', None)
        self._equivalent_plastic_strain = incremental_data.pop(
            'equivalent_plastic_strain', None)
        self._accumulated_shear_strain = incremental_data.pop(
            'accumulated_shear_strain', None
        )

        if incremental_data:
            unknown_fmt = ', '.join(['"{}"'.format(i) for i in incremental_data])
            msg = ('Unknown incremental data "{}". Allowed incremental data are: {}')
            raise ValueError(msg.format(unknown_fmt, self.allowed_data_fmt))

    @property
    def num_increments(self):
        return self._num_increments

    def __len__(self):
        return self.num_increments

    def __repr__(self):
        inc_data = []
        for i in self.incremental_data_names:
            inc_data.append(f'{i}{getattr(self, i).shape[1:] or (1,)}')
        out = (
            f'{self.__class__.__name__}('
            f'num_increments={len(self)}, '
            f'incremental_data={inc_data}'
            f')'
        )
        return out

    def to_dict(self):
        """Generate a dict representation."""

        return dict(self.incremental_data._asdict())

    @property
    def incremental_data_names(self):
        'Which of the allowed incremental data does this LoadResponse have?'
        has = []
        for allowed in self.ALLOWED_DATA:
            if getattr(self, allowed) is not None:
                has.append(allowed)
        return has

    @property
    def incremental_data(self):
        'Get all incremental data as a dict.'
        out = {i: None for i in LoadResponse.ALLOWED_DATA}
        for i in self.incremental_data_names:
            out.update({i: getattr(self, i)})
        return IncrementalData(**out)

    @staticmethod
    def required_incremental_data(cls, analysis_type=None):
        """Get a list of the required incremental data for a given type of analysis."""
        pass

    @property
    def allowed_data_fmt(self):
        'Get a comma-separated list of allowed incremental data.'
        return ', '.join(['"{}"'.format(i) for i in self.ALLOWED_DATA])

    @property
    def stress(self):
        return self._stress

    @property
    def equivalent_strain(self):
        return self._equivalent_strain

    @property
    def equivalent_plastic_strain(self):
        return self._equivalent_plastic_strain

    @property
    def accumulated_shear_strain(self):
        return self._accumulated_shear_strain

    @property
    @requires('stress')
    def principal_stress(self):
        """Get principal stress values."""
        return get_principal_values(self.stress)

    @requires('stress')
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
            source_dat, self.stress, value_idx=value_idx)

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
            yld_stress_principal = get_principal_values(yld_stress)
        else:
            yld_stress_principal = yld_stress

        return yld_stress_principal, good_idx

    @requires('stress')
    def is_uniaxial(self, increment=-1, tol=0.3):
        """Is the specified increment's true stress state approximately uniaxial?"""

        princ_stress = self.principal_stress[increment]

        # Principal values are ordered largest to smallest, so check the first is larger
        # than the other two:
        normed = princ_stress / princ_stress[0]
        uniaxial_factor = abs(normed[1:3])
        print(f'Checking if uniaxial with tolerance {tol}. Uniaxial factors are '
              f'{uniaxial_factor}.')
        if np.all((uniaxial_factor - tol) <= 0):
            return True
        else:
            return False


class LoadResponseSet(object):

    def __init__(self, load_responses):
        """

        Parameters
        ----------
        load_responses : list of (LoadResponse or dict)
            All LoadResponse objects within the list must have the same set of specified
            incremental data.

        """

        for idx, response in enumerate(load_responses):
            if not isinstance(response, LoadResponse):
                load_responses[idx] = LoadResponse(**response)

        # Validate:
        inc_data = load_responses[0].incremental_data_names
        for i in load_responses[1:]:
            if i.incremental_data_names != inc_data:
                msg = ('All load responses within a {} must have the same set of '
                       'specified incremental data.')
                raise InhomogeneousDataError(msg.format(self.__class__.__name__))

        self.responses = load_responses

        # All appended to in given methods, or in self.from_dict:
        self.yield_point_criteria = []      # `self.calculate_yield_stresses`
        self.yield_stresses = []            # `self.calculate_yield_stresses`
        self.fitted_yield_functions = []    # `self.calculate_yield_function_fit`

    def __repr__(self):
        inc_data = []
        for i in self.incremental_data_names:
            inc_data.append(f'{i}{getattr(self.responses[0], i).shape[1:] or (1,)}')
        out = (
            f'{self.__class__.__name__}('
            f'num_responses={len(self)}, '
            f'incremental_data={inc_data}'
            f')'
        )
        return out

    def __len__(self):
        return len(self.responses)

    def to_dict(self):
        """Generate a dict representation."""
        out = {
            'responses': [i.to_dict() for i in self.responses],
            'yield_point_criteria': [i.to_dict() for i in self.yield_point_criteria],
            'yield_stresses': [dict(i._asdict()) for i in self.yield_stresses],
            'fitted_yield_functions': [
                {
                    'YPC_idx': i['YPC_idx'],
                    'YPC_value_idx': i['YPC_value_idx'],
                    'yield_stress_idx': i['yield_stress_idx'],
                    'yield_function': {
                        'name': i['yield_function'].name,
                        **i['yield_function'].get_parameters(),
                        'fit_info': (dict(i['yield_function'].fit_info)
                                     if i['yield_function'].fit_info else None),
                        'yield_point': i['yield_function'].yield_point,
                    }
                }
                for i in self.fitted_yield_functions
            ],
        }
        return out

    @classmethod
    def from_dict(cls, dct):
        """Reconstitute a LoadResponseSet object from a dict."""

        obj = cls(load_responses=copy.deepcopy(dct['responses']))

        # Reconstitute a pre-existing `LoadResponseSet`:
        if 'yield_stresses' in dct:

            for yld_stress in dct['yield_stresses']:
                if not isinstance(yld_stress, YieldStresses):
                    yld_stress = YieldStresses(**yld_stress)
                    obj.yield_stresses.append(yld_stress)

            if 'yield_point_criteria' not in dct:
                msg = ('Yield stresses are defined according to one or more yield point '
                       'criteria, so `yield_point_criteria` must be given.')
                raise ValueError(msg)
            else:
                for ypc in dct['yield_point_criteria']:
                    if not isinstance(ypc, YieldPointCriteria):
                        ypc = YieldPointCriteria(**ypc)
                    obj.yield_point_criteria.append(ypc)

            if dct.get('fitted_yield_functions'):
                for yld_func in dct['fitted_yield_functions']:
                    if not isinstance(yld_func['yield_function'], YieldFunction):
                        yld_func_obj = YieldFunction.from_name(
                            **yld_func['yield_function']
                        )
                        yld_func_dict = {
                            'YPC_idx': yld_func['YPC_idx'],
                            'YPC_value_idx': yld_func['YPC_value_idx'],
                            'yield_stress_idx': yld_func['yield_stress_idx'],
                            'yield_function': yld_func_obj,
                        }
                    obj.fitted_yield_functions.append(yld_func_dict)

        return obj

    @property
    def incremental_data_names(self):
        """Which of the allowed LoadResponse incremental data do the constituent
        LoadResponse objects have?"""
        response = self.responses[0]
        has = []
        for allowed in response.ALLOWED_DATA:
            if getattr(response, allowed) is not None:
                has.append(allowed)
        return has

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

            num_excluded = 0
            for resp_idx, resp_i in enumerate(self.responses):

                val, _ = resp_i.get_yield_stress(yield_point_criteria, value_idx=ypv_idx)

                if val.size:
                    yield_stress_dict['values'].append(val)
                    yield_stress_dict['response_idx'].append(resp_idx)
                else:
                    num_excluded += 1

            yield_stress_dict['values'] = np.array(yield_stress_dict['values'])

            print(f'{num_excluded}/{len(self)} responses skipped for yield point '
                  f'{yield_point_val} from {yield_point_criteria}.')

            if not len(yield_stress_dict['values']):
                msg = (f'No yield stresses were found for yield point criteria '
                       f'value {yield_point_val}.')
                raise ValueError(msg)

            self.yield_stresses.append(YieldStresses(**yield_stress_dict))

        self.yield_point_criteria.append(yield_point_criteria)

    def _validate_yield_function_parameters(self, yield_function,
                                            yield_point_criteria_idx,
                                            yield_point_criteria_value_idx,
                                            uniaxial_response, **fixed_params):

        ypc = self.yield_point_criteria[yield_point_criteria_idx]

        uniaxial_eq_stress = None
        if 'equivalent_stress' in yield_function.PARAMETERS:

            if 'equivalent_stress' not in fixed_params:

                if uniaxial_response is not None:
                    # equivalent stress is calculated as the yield stress in the
                    # uniaxial response, for a given specified yield point:

                    if not isinstance(uniaxial_response, LoadResponse):
                        raise TypeError('`uniaxial_response` must be a LoadResponse')

                    if not uniaxial_response.is_uniaxial():
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
            if fixed_params.get('equivalent_stress') is not None:
                raise ValueError(msg.format('equivalent_stress'))

        fixed_params_copy = copy.deepcopy(fixed_params)
        if uniaxial_eq_stress:
            fixed_params_copy.update({
                'equivalent_stress': uniaxial_eq_stress,
            })

        return fixed_params_copy

    @at_most_one_of('equivalent_stress', 'uniaxial_response')
    def fit_yield_function(self, yield_function, yield_point_criteria_idx=None,
                           uniaxial_response=None, initial_params=None, opt_params=None,
                           **fixed_params):
        """Perform a fit to a yield function of all computed yield stresses.

        Parameters
        ----------
        yield_function : str or YieldFunction class
            The yield function to fit. Available yield functions can be displayed using:
                `from formable import get_available_yield_functions`
                `print(get_available_yield_functions())`
        yield_point_criteria_idx : list of (list of int of length 2), optional
            If specified, fit the yield function to only the yield stresses calculated
            from the specified yield point criteria value. By default, None, in which case
            the yield function is fitted to all available yield stresses.
        initial_params : dict, optional
            Any initial guesses for the fitting parameters. Mutually exclusive with
            additional keyword arguments passed, which are considered to be fixed.
        opt_params : dict, optional
            Optimisation parameters. Dict with any of the keys:
                default_bounds : list of length two, optional
                    The bounds applied to all non-fixed yield function parameters by
                    default.
                bounds : dict, optional
                    Dict of bounds for individual named parameters. These bounds take
                    precedence over `default_bounds`.
                **kwargs : dict
                    Other parameters to be passed to the SciPy least_squares function.
        **fixed_params : dict
            Any yield function parameters to be fixed during the fit can be passed as
            additional keyword arguments.

        """

        # Validation:
        YIELD_FUNCTION_MAP = get_yield_function_map()
        msg = 'Yield function "{}" not known. Available yield functions are: {}'
        available = ', '.join(['"{}"'.format(i) for i in YIELD_FUNCTION_MAP])
        msg = msg.format(yield_function, available)

        if isinstance(yield_function, str):
            if yield_function not in YIELD_FUNCTION_MAP:
                raise ValueError(msg)
            yield_function = YIELD_FUNCTION_MAP[yield_function]

        elif not issubclass(yield_function, YieldFunction):
            raise TypeError(msg)

        bad_kwargs = list(set(fixed_params.keys()) - set(yield_function.PARAMETERS))
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
                if ys.YPC_idx == ypc_idx and ys.YPC_value_idx == ypc_val_idx:
                    yield_stress = ys.values
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
                **fixed_params
            )

            # Perform fit:
            yld_func_obj = yield_function.from_fit(
                yield_stress,
                initial_params=initial_params,
                opt_params=opt_params,
                **updated_kwargs
            )

            ypc = self.yield_point_criteria[ypc_idx]
            yld_func_obj.yield_point = ypc.get_formatted(values_idx=ypc_val_idx)
            yield_func_dict['yield_function'] = yld_func_obj

            self.fitted_yield_functions.append(yield_func_dict)

    def remove_yield_function_fits(self):
        'Remove all yield function fits'

        self.fitted_yield_functions = []

    def show_yield_functions_3D(self, normalise=True, resolution=DEF_3D_RES,
                                equivalent_stress=None, min_stress=None, max_stress=None,
                                show_axes=True, planes=None, backend='plotly',
                                show_stress_states=True, join_stress_states=False,
                                show_contour_grid=False, layout=None,
                                include_yield_functions=None):
        'Visualise all fitted yield functions and data in 3D.'

        if not self.fitted_yield_functions:
            raise ValueError('No yield functions have been fitted to the load set.')

        yld_funcs = []
        yld_stresses = []
        stress_indices = []
        if not include_yield_functions:
            include_yield_functions = range(len(self.fitted_yield_functions))
        for yld_func_idx, yld_func_dict in enumerate(self.fitted_yield_functions):
            if yld_func_idx not in include_yield_functions:
                continue
            yld_funcs.append(yld_func_dict['yield_function'])
            yld_stress = self.yield_stresses[yld_func_dict['yield_stress_idx']]
            yld_stress_vals = yld_stress.values
            resp_idx = yld_stress.response_idx
            yld_stress_principal = get_principal_values(yld_stress_vals)
            yld_stresses.append(yld_stress_principal)
            stress_indices.append(resp_idx)

        if not show_stress_states:
            yld_stresses = None
            stress_indices = None

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
            layout=layout,
        )

    def show_yield_functions_2D(self, plane, normalise=True, resolution=DEF_2D_RES,
                                equivalent_stress=None, min_stress=None, max_stress=None,
                                show_axes=True, up=None, show_contour_grid=False,
                                show_stress_states=True, join_stress_states=False,
                                show_numerical_lankford=False,
                                show_numerical_lankford_fit=False, layout=None,
                                use_plotly_contour=False, include_yield_functions=None):
        'Visualise all fitted yield functions and data in 2D.'

        if not self.fitted_yield_functions:
            raise ValueError('No yield functions have been fitted to the load set.')

        yld_funcs = []
        yld_stresses = []
        stress_indices = []
        if not include_yield_functions:
            include_yield_functions = range(len(self.fitted_yield_functions))
        for yld_func_idx, yld_func_dict in enumerate(self.fitted_yield_functions):
            if yld_func_idx not in include_yield_functions:
                continue
            yld_funcs.append(yld_func_dict['yield_function'])
            yld_stress = self.yield_stresses[yld_func_dict['yield_stress_idx']]
            yld_stress_vals = yld_stress.values
            resp_idx = yld_stress.response_idx
            yld_stress_principal = get_principal_values(yld_stress_vals)
            yld_stresses.append(yld_stress_principal)
            stress_indices.append(resp_idx)

        if not show_stress_states:
            yld_stresses = None
            stress_indices = None

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
            show_numerical_lankford=show_numerical_lankford,
            show_numerical_lankford_fit=show_numerical_lankford_fit,
            layout=layout,
            use_plotly_contour=use_plotly_contour,
        )

    def show_yield_stresses_2D(self):
        pass


IncrementalData = namedtuple('IncrementalData', LoadResponse.ALLOWED_DATA)
YieldStresses = namedtuple(
    'YieldStresses', ['YPC_idx', 'YPC_value_idx', 'values', 'response_idx']
)
