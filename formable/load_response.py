"""`formable.load_response.py`"""

import copy
from pathlib import Path
from warnings import warn

import numpy as np
from scipy.optimize import least_squares
from damask_parse import read_table as read_damask_table

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
    def allowed_yield_point_types_fmt(self):
        'Get a comma-separated list of allowed incremental data.'
        return ', '.join(['"{}"'.format(i) for i in YIELD_POINT_FUNC_MAP])

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
        yield_stress : ndarray of shape (M, 3, 3)
            Yield stress tensors, one for each of M yield point criteria values.
        good_value_idx : list of int
            Indices of values for which the yield stress was successfully calculated.

        """

        source_dat = getattr(self, yield_point_criteria.source)
        yield_stress, good_value_idx = yield_point_criteria.get_yield_stress(
            source_dat, self.true_stress, value_idx=value_idx)

        return yield_stress, good_value_idx

    @init_yield_point_criteria
    def get_principal_yield_stress(self, yield_point_criteria):
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

        yld_stress = self.get_yield_stress(yield_point_criteria)
        yld_stress_principle, values_idx = get_principal_values(yld_stress)
        return yld_stress_principle, values_idx

    @requires('true_stress')
    def is_uniaxial(self, increment=-1, tol=1e-3):
        """Is the specified increment's true stress state approximately uniaxial?"""

        princ_stress = self.principal_true_stress[increment]

        # Principal values are ordered largest to smallest, so check the first is much
        # larger than the other two:
        normed = princ_stress / princ_stress[0]

        print('normed:\n{}\n'.format(normed))

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
        self.yield_functions = {}       # Updated in `self.calculate_yield_function_fit`

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

        yield_stresses = [{'values': [], 'response_idx': []}
                          for _ in yield_point_criteria.values]

        for resp_idx, resp_i in enumerate(self.responses):

            yld_stresses_i, vals_idx_i = resp_i.get_yield_stress(yield_point_criteria)

            for val_idx in vals_idx_i:
                yield_stresses[val_idx]['values'].append(yld_stresses_i[val_idx])
                yield_stresses[val_idx]['response_idx'].append(resp_idx)

        for i in yield_stresses:
            i['values'] = np.array(i['values'])

        self.yield_stresses.append(yield_stresses)
        self.yield_point_criteria.append(yield_point_criteria)

    def _validate_yield_function_parameters(self, yield_function, yield_point_criteria,
                                            uniaxial_response, **kwargs):

        uniaxial_eq_stresses = None
        if 'equivalent_stress' in yield_function.PARAMETERS:

            if 'equivalent_stress' not in kwargs:

                if uniaxial_response is not None:
                    # equivalent stress is calculated as the yield stress in the
                    # uniaxial response, for a given specified yield point:

                    if not isinstance(uniaxial_response, LoadResponse):
                        raise TypeError('`uniaxial_response` must be a LoadResponse')

                    if not uniaxial_response.is_uniaxial(tol=1e-3):
                        msg = ('Specified `uniaxial_response` does not appear to be '
                               'uniaxial.')
                        raise ValueError(msg)

                        princ_stress, _ = uniaxial_response.get_principal_yield_stress(
                            yield_point_criteria)

                        if len(princ_stress) != len(yield_point_criteria):
                            msg = ('Yield point not reached within uniaxial response.')
                            raise ValueError(msg)

                        # Turn into scalars:
                        uniaxial_eq_stresses = [i[0] for i in princ_stress]

        else:

            msg = (f'The yield function {yield_function.__name__} does not require '
                   'an equivalent stress parameter, so `{}` is not required.')
            if uniaxial_response is not None:
                raise ValueError(msg.format('uniaxial_response'))
            if kwargs.get('equivalent_stress') is not None:
                raise ValueError(msg.format('equivalent_stress'))

        kwargs_split = []
        for idx, i in enumerate(yield_point_criteria.values):
            kwargs_i = copy.deepcopy(kwargs)
            if uniaxial_eq_stresses:
                kwargs_i.update({
                    'equivalent_stress': uniaxial_eq_stresses[idx]
                })
            kwargs_split.append(kwargs_i)

        return kwargs_split

    @at_most_one_of('equivalent_stress', 'uniaxial_response')
    def fit_yield_function(self, yield_function, equivalent_strain=None,
                           uniaxial_response=None, **kwargs):
        """Perform a fit to a yield function of all computed yield stresses.

        Parameters
        ----------
        yield_function : str or YieldFunction class
            The yield function to fit. Available yield functions can be displayed using: 
                `from formable import AVAILABLE_YIELD_FUNCTIONS`
                `print(AVAILABLE_YIELD_FUNCTIONS)`

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

        yld_func_name = yield_function.__name__
        if yld_func_name not in self.yield_functions:
            self.yield_functions.update({yld_func_name: []})

        for ypc_idx, ypc in enumerate(self.yield_point_criteria):

            kwargs = self._validate_yield_function_parameters(
                yield_function, ypc, uniaxial_response, **kwargs)

            yld_func_list = []
            for ypc_val_idx, yld_stress in enumerate(self.yield_stresses[ypc_idx]):

                # Perform fit:
                yld_func_obj = yield_function.from_fit(yld_stress, **kwargs[ypc_val_idx])
                yld_func_list.append(yld_func_obj)

            self.yield_functions[yld_func_name].append(yld_func_list)

    def show_yield_functions_3D(self, equivalent_strain=None, normalise=True,
                                resolution=DEF_3D_RES, equivalent_stress=None,
                                min_stress=None, max_stress=None, show_axes=True,
                                planes=None, backend='plotly', **kwargs):
        'Visualise all fitted yield functions and data in 3D.'

        _, yield_point_type, yield_point_value = kwargs['yield_criteria']
        yield_point_tup = (yield_point_type, yield_point_value)

        # TODO: pass flatten yield functions and stress states lists into 1D
        # yield functions need to have a yield point criteria associated with them (maybe
        # just as a formatted string) so it can appear in the legend.

        if yield_point not in self.yield_stresses:
            msg = 'Yield point {} has not been fitted.'
            raise ValueError(msg.format(yield_point))

        yld_point_dct = {yield_point[0]: yield_point[1]}
        stress_states = self.get_principal_yield_stresses(**yld_point_dct)
        yield_functions = [i[yield_point] for i in self.yield_functions.values()]

        return YieldFunction.compare_3D(
            yield_functions,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            show_axes=show_axes,
            planes=planes,
            stress_states=stress_states,
            backend=backend,
        )

    def show_yield_functions_2D(self, yield_point, plane, normalise=True,
                                resolution=DEF_2D_RES, equivalent_stress=None,
                                min_stress=None, max_stress=None, show_axes=True,
                                up=None, show_contour_grid=False):
        'Visualise all fitted yield functions and data in 2D.'

        # TODO: pass multiple stress states and equivalent stresses based on all fitted
        # yield points?

        if yield_point not in self.yield_stresses:
            msg = 'Yield point {} has not been fitted.'
            raise ValueError(msg.format(yield_point))

        yld_point_dct = {yield_point[0]: yield_point[1]}
        stress_states = self.get_principal_yield_stresses(**yld_point_dct)
        yield_functions = [i[yield_point] for i in self.yield_functions.values()]

        return YieldFunction.compare_2D(
            yield_functions,
            plane,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            stress_states=stress_states,
            up=None,
            show_contour_grid=False,
        )
