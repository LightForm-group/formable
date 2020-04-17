"""`formable.utilities.py`"""

import functools
import numpy as np


def requires(incremental_data_name):

    def requires_wrapped(func):

        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            if getattr(self, incremental_data_name) is None:
                msg = (f'{self.__class__.__name__} must have "{incremental_data_name}" '
                       'incremental data to do this. Re-instantiate with this data.')
                raise ValueError(msg)
            return func(self, *args, **kwargs)

        return wrapped

    return requires_wrapped


def at_most_one_of(*conditioned_args):

    def requires_wrapped(func):

        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):

            num_specified = sum([i in kwargs for i in conditioned_args])
            if num_specified > 1:
                msg = 'Specify at most one of these keyword arguments: {}'
                cond_args_fmt = ', '.join(['"{}"'.format(i) for i in conditioned_args])
                raise ValueError(msg.format(cond_args_fmt))
            return func(self, *args, **kwargs)

        return wrapped

    return requires_wrapped


def format_axis_label(direction, basis_labels=['x', 'y', 'z'], zero_tol=1e-6):
    out = []
    for i, lab in zip(direction, basis_labels):
        if abs(i) < zero_tol:
            continue

        if i == 1:
            i_fmt = ''
        elif i == -1:
            i_fmt = '-'
        else:
            i_fmt = '{:.4g}'.format(i)
        out.append(i_fmt + lab)

    out_fmt = ' + '.join(out)

    return out_fmt


def parse_rgb_str(rgb_str):
    'Get the list of integers in an RGB color string like "rgb(xxx, xxx, xxx)"'
    return [int(i) for i in rgb_str[4:-1].split(',')]


def format_rgba(*rgb_comps, opacity=1):
    'Get an "rgba(xxx, xxx, xxx, xxx)" string.'
    return 'rgba({0},{1},{2},{3})'.format(*rgb_comps, opacity)


def expand_idx_array(arr, max_idx):
    """Expand an index array into a masked array where an element at position
    i has value i, for values drawn from the original array, and all other
    elements are masked.

    Parameters
    ----------
    arr : ndarray of shape (N,)
    max_idx : int
        Maximum index to consider. The output arrays will
        have length `max_idx` + 1.

    Returns
    -------
    full : MaskedArray of shape (`max_idx` + 1,)
        Masked array where unmasked values at position i have value i.
    idx : MaskedArray of shape (`max_idx` + 1,)
        Masked array giving the positions in the original array of the
        elements in `full`.

    Example
    -------
    >>> A = np.array([1, 3, 4, 0])
    >>> full, idx = expand_idx_array(A, max_idx=4)
    >>> print(full)
    [0 1 -- 3 4]
    >>> print(idx)
    [3 0 -- 1 2]

    """

    full = np.ma.masked_all((max_idx + 1,), dtype=int)
    idx = np.ma.masked_all((max_idx + 1,), dtype=int)
    full[arr] = arr
    idx[arr] = np.arange(len(arr))

    return full, idx
