"""`formable.load_response.py`

TODO:
    - This needs generalising.
        - The contained classes should be useful for fitting yield functions to
          experimental tests as well as simulated tests.
        - This package should not depend on `damask-parse`.

"""

from pathlib import Path
from warnings import warn

import numpy as np
from scipy.optimize import least_squares
from damask_parse import read_table as read_damask_table

from formable.yield_functions import (
    YieldFunction, YIELD_FUNCTION_MAP, DEF_3D_RES, DEF_2D_RES)


class LoadResponse(object):

    def __init__(self, path, load_set):
        """Parse the response file."""
        response = read_damask_table(path)
        self.load_set = load_set
        for key, val in response.items():
            key_san = key.replace('(', '_').replace(')', '_').lower()
            setattr(self, key_san, val)

    def get_yield_stress(self, yield_point, yield_point_column_name):
        """Find the yield stress associated with a yield_point value."""

        yield_point_column = getattr(self, yield_point_column_name)

        # Linearly interpolate yield stress
        hi_idx = np.where(abs(yield_point_column) >= yield_point)[0][0]
        lo_idx = hi_idx - 1
        hi_yld = abs(yield_point_column[hi_idx])
        lo_yld = abs(yield_point_column[lo_idx])

        lo_stress_factor = (hi_yld - yield_point) / (hi_yld - lo_yld)
        hi_stress_factor = (yield_point - lo_yld) / (hi_yld - lo_yld)

        yld_stress = (
            getattr(self, 'cauchy')[lo_idx] * lo_stress_factor +
            getattr(self, 'cauchy')[hi_idx] * hi_stress_factor
        )

        return yld_stress

    def get_principle_yield_stress(self, yield_point, yield_point_column_name):
        yld_stress = self.get_yield_stress(yield_point, yield_point_column_name)
        out = np.linalg.eigvals(yld_stress)

        return out

    def validate(self, *args):
        """Validate these results against a set of required results."""
        for attr in args:
            if not hasattr(self, attr):
                return False
        return True

    @property
    def principle_stress(self):
        if hasattr(self, 'cauchy'):
            return np.linalg.eigvals(getattr(self, 'cauchy'))


class LoadResponseSet(object):

    def __init__(self, path, grid_size, num_grains, inc, time, uniaxial_path):
        """
        Parameters
        ----------
        path : Path object
            Path to a parent directory of sub directories that contain
            files from simulations of distinct load cases.
        """

        self.grid_size = grid_size
        self.num_grains = num_grains
        self.inc = inc
        self.time = time

        # Yield function fits for given "yield point" conditions, assigned in
        # `add_yield_function_fit`:
        self.yield_functions = {}

        self.yield_stresses = {}
        self.uniaxial_yield_stress = {}

        self.skipped = []
        self.skipped_uniaxial = []

        self.responses = []
        self.uniaxial_response = None

        uniaxial_path = Path(uniaxial_path)
        if not uniaxial_path.is_dir():
            raise ValueError('`unaxial_path` is not a directory.')

        uni_out_path = uniaxial_path.joinpath('geom_load.txt')
        if not uni_out_path.is_file():
            if uniaxial_path.joinpath('postProc').is_dir():
                out_path = uniaxial_path.joinpath('postProc', 'geom_load.txt')
            else:
                self.skipped_uniaxial.append(uniaxial_path)

        self.uniaxial_response = LoadResponse(uni_out_path, self)

        path = Path(path)
        if not path.is_dir():
            raise ValueError('`path` is not a directory.')

        for i in path.glob('*'):

            if not i.is_dir():
                continue

            out_path = i.joinpath('geom_load.txt')
            if not out_path.is_file():
                if i.joinpath('postProc').is_dir():
                    out_path = i.joinpath('postProc', 'geom_load.txt')
                else:
                    self.skipped.append(i)
                    continue

            self.responses.append(LoadResponse(out_path, self))

    def compute_yield_stresses(self, yield_points):
        """Compute yield stresses for each load case, for multiple yield points.

        Parameters
        ----------
        yield_points : list of tuple of (str, number)
            List of yield point definitions at which to fit the yield function. A yield
            point definition as two parts: the first is the type of yield point. This
            must be one of "equivalent_strain" or "total_shear". The second is the value
            of this quantity at which we are to consider yield to occur.

        """

        for yld_point in yield_points:
            yld_point_type, yld_point_val = yld_point

            if yld_point in self.yield_stresses:
                msg = ('Yield point {} has already been calculated.')
                raise ValueError(msg.format(yld_point))

            self.yield_stresses[yld_point] = self._get_yield_data(
                yld_point_val, yld_point_type)
            self.uniaxial_yield_stress[yld_point] = self._get_uniaxial_yield_data(
                yld_point_val, yld_point_type)

    def get_principle_yield_stresses(self, equivalent_strain=None, total_shear=None):

        # Specify one yield point type only:
        if (equivalent_strain is not None and total_shear is not None) or (
                equivalent_strain is None and total_shear is None):
            msg = 'Specify exactly one of "equivalent_strain" and "total_shear".'
            raise ValueError(msg)

        if not self.yield_stresses:
            return None

        msg = 'Yield point {} has not been calculated.'
        if equivalent_strain is not None:
            yld_point = ('equivalent_strain', equivalent_strain)
            if yld_point not in self.yield_stresses:
                raise ValueError(msg.format(yld_point))

        elif total_shear is not None:
            yld_point = ('total_shear', total_shear)
            if yld_point not in self.yield_stresses:
                raise ValueError(msg.format(yld_point))

        principals = np.linalg.eigvals(self.yield_stresses[yld_point])
        principals = np.sort(principals, axis=1)[:, ::-1]

        return principals

    def add_yield_function_fit(self, yield_function, yield_points=None, **kwargs):
        """Perform a fit of the stress data to a yield function.

        Parameters
        ----------
        yield_function : str or YieldFunction class
            The yield function to fit. Available yield functions can be displayed using: 
                `from formable import yield_functions;`
                `print(yield_functions.AVAILABLE_YIELD_FUNCTIONS)`
        yield_points : list of tuple of (str, number), optional
            List of yield point definitions at which to fit the yield function. A yield
            point definition as two parts: the first is the type of yield point. This
            must be one of "equivalent_strain" or "total_shear". The second is the value
            of this quantity at which we are to consider yield to occur. If `None`, fit to
            any yield stresses already computed.
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

        elif not isinstance(yield_function, YieldFunction):
            raise TypeError(msg)

        # Compute yield stresses for any yield points for which stresses have not been
        # calculated:
        to_add = []
        if yield_points:
            to_add = list(set(self.yield_stresses.keys()) - set(yield_points))
            if to_add:
                self.compute_yield_stresses(to_add)

        yield_points = (yield_points or []) + list(self.yield_stresses.keys())

        if not self.yield_stresses:
            msg = 'Specify some yield points at which to fit the yield function.'
            raise ValueError(msg)

        for yld_point in yield_points:

            yld_stress = self.yield_stresses[yld_point]

            if 'equivalent_stress' in yield_function.PARAMETERS:
                if 'equivalent_stress' not in kwargs:
                    kwargs.update({
                        'equivalent_stress': self.uniaxial_yield_stress[yld_point]
                    })

            # Fit:
            yld_func_obj = yield_function.from_fit(yld_stress, **kwargs)

            # Store the fitted yield function according to the "yield point" definition:
            yld_func_name = yield_function.__name__
            if yld_func_name not in self.yield_functions:
                self.yield_functions.update({yld_func_name: {}})

            self.yield_functions[yld_func_name][yld_point] = yld_func_obj

    def _get_uniaxial_yield_data(self, yield_point, yield_point_type='total_shear'):

        if yield_point_type not in ['total_shear', 'equivalent_strain']:
            msg = (f'yield_point_type` must be either "total_shear" or '
                   f'"equivalent_strain" not "{yield_point_type}".')
            raise ValueError(msg)

        yield_point_col_name_opts = {
            'total_shear': 'totalshear',
            'equivalent_strain': 'mises_ln_v__',
        }
        yield_point_col_name = yield_point_col_name_opts[yield_point_type]

        resp = self.uniaxial_response
        yield_stress = resp.get_yield_stress(yield_point, yield_point_col_name)

        return yield_stress[2, 2]

    def _get_yield_data(self, yield_point, yield_point_type='total_shear'):
        """Get yield point data."""

        if yield_point_type not in ['total_shear', 'equivalent_strain']:
            msg = (f'yield_point_type` must be either "total_shear" or '
                   f'"equivalent_strain" not "{yield_point_type}".')
            raise ValueError(msg)

        yield_point_col_name_opts = {
            'total_shear': 'totalshear',
            'equivalent_strain': 'mises_ln_v__',
        }
        yield_point_col_name = yield_point_col_name_opts[yield_point_type]

        yield_stress = []
        skipped_responses = []
        for idx, resp in enumerate(self.responses):

            if not resp.validate('cauchy', yield_point_col_name):
                continue
            try:
                ys_i = resp.get_yield_stress(yield_point, yield_point_col_name)
                yield_stress.append(ys_i)
            except IndexError:
                skipped_responses.append(idx)

        yield_point_type_fmt = f'{yield_point_type:>20s}'
        skipped_fmt = f'{len(skipped_responses)}/{len(self.responses)}'
        print(f'Yield point: {yield_point}, {yield_point_type_fmt}; '
              f' {skipped_fmt:>7s} responses skipped.')

        return np.array(yield_stress)

    def show_yield_functions_3D(self, yield_point, normalise=True, resolution=DEF_3D_RES,
                                equivalent_stress=None, min_stress=None, max_stress=None,
                                show_axes=True, planes=None, backend='plotly'):
        'Visualise all fitted yield functions and data in 3D.'

        # TODO: pass multiple stress states and equivalent stresses based on all fitted
        # yield points?

        if yield_point not in self.yield_stresses:
            msg = 'Yield point {} has not been fitted.'
            raise ValueError(msg.format(yield_point))

        yld_point_dct = {yield_point[0]: yield_point[1]}
        stress_states = self.get_principle_yield_stresses(**yld_point_dct)
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
        stress_states = self.get_principle_yield_stresses(**yld_point_dct)
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
