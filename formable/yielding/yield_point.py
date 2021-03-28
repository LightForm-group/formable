"""`formable.yielding.yield_stress.py`

Module containing:
    - functions that calculate yield stress according to some yield point criterion.
    - a class to represent yield point criteria of a given type

"""

import functools

import numpy as np


class YieldPointUnsatisfiedError(Exception):
    pass


def get_yield_stress_from_equivalent_strain(equivalent_strain, yield_point, true_stress):
    """Interpolate the yield stress based on a Von Mises equivalent strain yield point.

    Parameters
    ----------
    true_stress : ndarray of shape (N, 3, 3)
        True stress states for each of N increments.
    equivalent_strain : ndarray of shape (N,)
        Von Mises equivalent strain for each of N increments.
    yield_point : float
        Von Mises equivalent strain at which yield stress is to be interpolated.

    Returns
    -------
    yield_stress : ndarray of shape (3, 3)
        Interpolated stress state at which Von Mises equivalent strain is equal to the
        specified `yield_point`.

    """

    try:
        hi_idx = np.where(abs(equivalent_strain) >= yield_point)[0][0]
        if hi_idx == 0:
            raise IndexError
    except IndexError:
        msg = 'Equivalent strain "{}" is not achieved within the specified increments.'
        raise YieldPointUnsatisfiedError(msg.format(yield_point))

    lo_idx = hi_idx - 1
    hi_yld = abs(equivalent_strain[hi_idx])
    lo_yld = abs(equivalent_strain[lo_idx])

    lo_stress_factor = (hi_yld - yield_point) / (hi_yld - lo_yld)
    hi_stress_factor = (yield_point - lo_yld) / (hi_yld - lo_yld)

    yld_stress = (
        true_stress[lo_idx] * lo_stress_factor +
        true_stress[hi_idx] * hi_stress_factor
    )

    return yld_stress


class YieldPointCriteria(object):

    # Mapping (source, threshold) to the function that return the yield stress:
    YIELD_POINT_FUNC_MAP = {
        ('equivalent_strain',
         'equivalent_strain'): get_yield_stress_from_equivalent_strain,
        ('equivalent_plastic_strain',
         'equivalent_plastic_strain'): get_yield_stress_from_equivalent_strain,
        ('accumulated_shear_strain',
         'accumulated_shear_strain'): get_yield_stress_from_equivalent_strain,
    }

    def __init__(self, threshold, values, source=None, **kwargs):

        values = np.array(values if isinstance(values, list) else [values])

        self.threshold = threshold
        self.values = values
        self.source = source or threshold
        self.kwargs = kwargs

        # TODO: source must be an allowed LoadResponse incremental_data

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'threshold={self.threshold!r}, '
                f'values={self.values!r}, '
                f'source={self.source!r})')

    def to_dict(self):
        """Generate a dict representation."""
        out = {
            'threshold': self.threshold,
            'values': self.values,
            'source': self.source,
            **self.kwargs,
        }
        return out

    def get_formatted(self, values_idx=None):
        if values_idx:
            if values_idx > (len(self) - 1):
                msg = (f'YieldPointCriteria has only {len(self)} value(s); cannot '
                       f'format `values_idx={values_idx}`.')
                raise IndexError(msg)
        if values_idx is None:
            return f'{self.threshold} = {self.values}'
        else:
            return f'{self.threshold} = {self.values[values_idx]}'

    def get_yield_stress(self, source_data, stress_data, value_idx=None):
        """
        Parameters
        ----------
        source_data : ndarray of outer shape (N,)
            Incremental data used as the source data to determine the yield point.
        stress_data : ndarray of shape (N, 3, 3)
            Stress tensors from which a yield stress can be calculated
        value_idx : int, optional
            If specified, the yield stress for only this value of the yield point criteria
            will be calculated. If not specified, yield stresses for all yield point
            criteria values will be calculated.

        Returns
        -------
        yield_stress : ndarray of float of shape (M, 3, 3) or (3, 3)
            Yield stress tensors, one for each of M yield point criteria values for which
            the yield stress was successfully calculated. M therefore ranges from 0 to
            the number of values in the yield point criteria. If `value_idx` is specified
            and the yield stress is successfully calculated, the shape will be (3, 3).
        good_idx : ndarray of int of shape (M,)
            Indices of the M values for which the yield stress was successfully
            calculated. If none were successfully calculated, this list will be empty.

        """

        func = self.YIELD_POINT_FUNC_MAP[(self.source, self.threshold)]

        if value_idx is None:
            value_idx_list = list(range(len(self)))
        else:
            value_idx_list = [value_idx]

        good_idx = []
        yield_stress = []

        for value_idx_i in value_idx_list:
            value = self.values[value_idx_i]
            try:
                yld_str_i = func(source_data, value, stress_data, **self.kwargs)
            except YieldPointUnsatisfiedError:
                continue

            good_idx.append(value_idx_i)
            yield_stress.append(yld_str_i)

        yield_stress = np.array(yield_stress).reshape(-1, 3, 3)
        good_idx = np.array(good_idx)

        if (value_idx is not None) and yield_stress.size:
            yield_stress = yield_stress[0]

        return yield_stress, good_idx


def init_yield_point_criteria(func):

    @functools.wraps(func)
    def wrapped(self, yield_point_criteria, *args, **kwargs):

        if not isinstance(yield_point_criteria, YieldPointCriteria):
            yield_point_criteria = YieldPointCriteria(**yield_point_criteria)

        return func(self, yield_point_criteria, *args, **kwargs)

    return wrapped
