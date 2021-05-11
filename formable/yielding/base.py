"""`formable.yielding.base.py`

Contains an abstract base class to represent a yield function. All yield function classes
inherit from this class.

"""

import abc
import copy
import functools

import numpy as np
from plotly import graph_objects
from plotly.colors import DEFAULT_PLOTLY_COLORS
from scipy.optimize import least_squares, curve_fit, OptimizeResult
from skimage.measure import find_contours
from ipywidgets import widgets

from formable import utils, maths_utils
from formable.yielding.map import get_yield_function_map

DEF_3D_RES = 50
DEF_2D_RES = 100


def yield_function_fitter(func):
    """Decorator for the `residual` static method of each YieldFunction subclass.

    This decorator ensures all yield function parameters are contained within the `kwargs`
    dict, so they can be uniformly accessed within the `residual` function (regardless of
    which parameters are being fitted and which are static).

    """
    @functools.wraps(func)
    def inner(fitting_params, stress_states, fitting_param_names, **kwargs):
        params = dict(zip(fitting_param_names, fitting_params))
        kwargs.update(**params)
        return func(fitting_params, stress_states, fitting_param_names, **kwargs)

    return inner


class YieldFunction(metaclass=abc.ABCMeta):

    PARAMETERS = []
    LITERATURE_VALUES = {}
    FORMATTED_PARAMETER_NAMES = {}

    def __repr__(self):
        params_fmt = ['{}={:.4f}'.format(k, v)
                      for k, v in self.get_parameters(formatted=False).items()]
        out = (
            f'{self.__class__.__name__}(\n\t' +
            ',\n\t'.join(params_fmt) +
            f'\n)'
        )
        return out

    def __str__(self):

        param_names = []
        param_vals = []
        for i, j in self.get_parameters(formatted=True).items():
            param_names.append(i)
            param_vals.append(j)

        max_char_len = max([len(i) for i in param_names])
        params_fmt = [
            '{{:>{}}} = {{:.4f}}'.format(max_char_len).format(k, v)
            for k, v in zip(param_names, param_vals)
        ]

        out = (
            f'{self.__class__.__name__}\n' +
            '\n'.join(params_fmt)
        )
        return out

    @staticmethod
    @abc.abstractmethod
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):
        pass

    def get_value(self, stress_states):

        value = self.residual(
            fitting_params=[],
            stress_states=stress_states,
            fitting_param_names=[],
            **self.get_parameters(formatted=False)
        )
        return value

    @classmethod
    def from_name(cls, name, fit_info=None, yield_point=None, **parameters):
        'Get a specific yield function from the name and parameters.'
        YIELD_FUNCTION_MAP = get_yield_function_map()
        if name not in YIELD_FUNCTION_MAP:
            available = ', '.join([f'{i!r}' for i in YIELD_FUNCTION_MAP])
            msg = (f'Yield function "{name}" not known. Available yield functions '
                   f'are: {available}.')
            raise ValueError(msg)

        if fit_info:
            fit_info = OptimizeResult(**fit_info)

        yld_func_class = YIELD_FUNCTION_MAP[name]
        yld_func = yld_func_class(**parameters)
        yld_func.fit_info = fit_info
        yld_func.yield_point = yield_point

        return yld_func

    @classmethod
    def from_fit(cls, stress_states, initial_params=None, force_fit=False,
                 opt_params=None, **fixed_params):
        """Fit the yield function to yield stress states.

        Parameters
        ----------
        stress_states : ndarray of shape (N, 3, 3)
        initial_params : dict
            Any initial guesses for the fitting parameters. Mutually exclusive with
            additional keyword arguments (**fixed_params) passed, which are considered
            to be fixed.
        force_fit : bool, optional
            If False, do not fit if there is insufficient input data. False by default.
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
            Additional parameters passed to this method will be fixed during the fit.

        Returns
        -------
        yield_function : YieldFunction
            The fitted yield function, with a populated `fit_info` attribute if a fit
            was performed.

        """

        fitting_param_names = [i for i in cls.PARAMETERS if i not in fixed_params]
        initial_params_all = np.ones(len(fitting_param_names))

        initial_params = initial_params or {}

        for param_name, initial_guess in initial_params.items():
            if param_name not in fitting_param_names:
                if param_name not in cls.PARAMETERS:
                    msg = (f'Initial guess specified for parameter "{param_name}", but '
                           f'this parameter does not exist for the specified yield '
                           f'function, "{cls.__name__}".')
                    raise ValueError(msg)
                elif param_name in fixed_params:
                    msg = (f'Initial guess specified for parameter "{param_name}", but '
                           f'this parameter has also been specified as a keyword '
                           f'argument, indicating it should be fixed.')
                    raise ValueError(msg)

            param_idx = fitting_param_names.index(param_name)
            initial_params_all[param_idx] = initial_guess

        if not fitting_param_names:

            # Construct yield function, but no need to fit:
            yield_function = cls(**fixed_params)
            fit_info = None

        else:

            if len(fitting_param_names) > len(stress_states) and not force_fit:
                msg = (f'Insufficient number of stress states ({len(stress_states)}) to '
                       f'fit yield function "{cls.__name__}" with '
                       f'{len(fitting_param_names)} fitting parameters.')
                raise ValueError(msg)

            # Set something sensible for the initial equivalent stress, if it is a
            # fitting parameter, and not given an initial guess:
            if ('equivalent_stress' in fitting_param_names and
                    'equivalent_stress' not in initial_params):
                idx = fitting_param_names.index('equivalent_stress')
                initial_params_all[idx] = 50e6

            bounds_all = (-np.inf, +np.inf)
            if opt_params:
                opt_params = copy.deepcopy(opt_params)
                def_bounds = opt_params.pop('default_bounds', None)
                bounds_dict = opt_params.pop('bounds', None)

                if def_bounds:
                    bounds_all = [
                        [def_bounds[0]] * len(initial_params_all),
                        [def_bounds[1]] * len(initial_params_all),
                    ]
                else:
                    bounds_all = [
                        [-np.inf] * len(initial_params_all),
                        [+np.inf] * len(initial_params_all),
                    ]

                if bounds_dict:
                    for param_name, param_bounds in bounds_dict.items():

                        if param_name not in cls.PARAMETERS:
                            msg = (f'Fitting bounds specified for parameter '
                                   f'"{param_name}", but this parameter does not exist '
                                   f'for the specified yield function, "{cls.__name__}".')
                            raise ValueError(msg)
                        elif param_name in fixed_params:
                            msg = (f'Fitting bounds specified for parameter '
                                   f'"{param_name}", but this parameter has also been '
                                   f'specified as a keyword argument, indicating it '
                                   f'should be fixed.')
                            raise ValueError(msg)

                        param_idx = fitting_param_names.index(param_name)
                        bounds_all[0][param_idx] = param_bounds[0]
                        bounds_all[1][param_idx] = param_bounds[1]

            fit_info = least_squares(
                cls.residual,
                initial_params_all,
                bounds=tuple(bounds_all),
                kwargs=dict(
                    stress_states=stress_states,
                    fitting_param_names=fitting_param_names,
                    **fixed_params,
                ),
                **(opt_params or {}),
            )

            parameters = dict(zip(fitting_param_names, fit_info.x))
            parameters.update(**fixed_params)
            yield_function = cls(**parameters)

        yield_function.fit_info = fit_info

        return yield_function

    @classmethod
    def from_literature(cls, material, reference, **kwargs):

        if material not in cls.LITERATURE_VALUES:
            avail_mats_fmt = ', '.join(['"{}"'.format(i)
                                        for i in cls.LITERATURE_VALUES.keys()])
            if not avail_mats_fmt:
                avail_mats_fmt = 'None'
            msg = ('Materials "{}" not found in recorded literature values for yield'
                   'function "{}". Maybe you can add it? Recorded materials are: {}.')
            raise ValueError(msg.format(material, cls.__name__, avail_mats_fmt))

        if reference not in cls.LITERATURE_VALUES[material]:
            avail_refs_fmt = ', '.join(['"{}"'.format(i)
                                        for i in cls.LITERATURE_VALUES[material].keys()])
            if not avail_refs_fmt:
                avail_refs_fmt = 'None'
            msg = ('Refernce "{}" not found in recorded literature values. Recorded '
                   'references for materials "{}" are {}.')
            raise ValueError(msg.format(reference, material, avail_refs_fmt))

        return cls(**cls.LITERATURE_VALUES[material][reference]['parameters'], **kwargs)

    def get_parameters(self, formatted=False, format_type='unicode'):
        'Get a dict of the parameters defining this yield surface.'

        out = {k: getattr(self, k) for k in self.PARAMETERS}

        if (formatted and self.FORMATTED_PARAMETER_NAMES and
                format_type in self.FORMATTED_PARAMETER_NAMES):
            out = {
                self.FORMATTED_PARAMETER_NAMES[format_type].get(k, k): v
                for k, v in out.items()
            }

        return out

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def compare_3D(cls, yield_functions, normalise=True, resolution=DEF_3D_RES,
                   equivalent_stress=None, min_stress=None, max_stress=None,
                   show_axes=True, planes=None, stress_states=None, backend='plotly',
                   stress_indices=None, join_stress_states=False, show_contour_grid=False,
                   legend_text=None, layout=None):
        """Visualise one or more yield functions in 3D.

        Parameters
        ----------
        yield_functions : list of YieldFunction
            List of YieldFunction objects to display. If a given yield function has an
            attribute `yield_point` assigned, this will be displayed in the legend.
        stress_states : list of ndarray, optional
            List of principal stress states to show. If specified, the length of this list
            must match that of `yield_functions`. In that case, if there are no stress
            states to show for one of more of the passed yield functions, then those list
            elements should be `None`.

        """

        if stress_states is not None:
            if len(stress_states) != len(yield_functions):
                msg = ('If specified, `stress_states` should be a list of equal length '
                       'to the `yield_functions` list.')
                raise ValueError(msg)

            if join_stress_states and not stress_indices:
                msg = ('If `join_stress_states=True`, `stress_indices` must be specified '
                       'as a list (of equal length to `stress_states`) of integer '
                       'indices that determine how the stress states should be '
                       'connected.')
                raise ValueError(msg)

            if stress_indices and len(stress_indices) != len(stress_states):
                msg = '`stress_indices` must have the same length as `stress_states`.'
                raise ValueError(msg)

        if legend_text:
            if len(legend_text) != len(yield_functions):
                msg = ('If specified, `legend_text` must be a list of equal length to '
                       '`yield_function`.')
                raise ValueError(msg)

        stress_range = cls._get_multi_plot_stress_range(
            yield_functions=yield_functions,
            normalise=normalise,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
        )
        min_stress = stress_range['min_stress']
        max_stress = stress_range['max_stress']
        eq_stress = stress_range['eq_stress']

        (stress_X, stress_Y, stress_Z), values_all = cls.get_3D_multi_plot_data(
            yield_functions,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=eq_stress,
            min_stress=min_stress,
            max_stress=max_stress,
        )

        if backend == 'pyvista':

            raise NotImplementedError('not yet.')

        elif backend == 'plotly':

            stress_X = stress_X.flatten()
            stress_Y = stress_Y.flatten()
            stress_Z = stress_Z.flatten()

            fig_data = []

            if show_contour_grid:

                min_x, max_x = min(stress_X), max(stress_X)
                min_y, max_y = min(stress_Y), max(stress_Y)
                min_z, max_z = min(stress_Z), max(stress_Z)

                fig_data.append({
                    'type': 'scatter3d',
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'x',
                        'color': 'black',
                        'size': 4,
                    },
                    'x': [min_x, min_x, min_x, min_x, max_x, max_x, max_x, max_x],
                    'y': [min_y, min_y, max_y, max_y, min_y, min_y, max_y, max_y],
                    'z': [min_z, max_z, min_z, max_z, min_z, max_z, min_z, max_z],
                })

            for idx, (yld_func, i) in enumerate(zip(yield_functions, values_all)):

                name = f'{idx + 1}. {yld_func.__class__.__name__}'
                yield_point = getattr(yld_func, 'yield_point', None)
                if legend_text:
                    name += f' [{legend_text[idx]}] '
                if yield_point:
                    name += f' ({yield_point})'

                fig_data.append({
                    'type': 'isosurface',
                    'x': stress_X,
                    'y': stress_Y,
                    'z': stress_Z,
                    'value': i,
                    'name': name,
                    'opacity': 0.5,
                    'showlegend': True,
                    'isomin': 0,
                    'isomax': 0,
                    'surface_count': 1,
                    'colorscale': [
                        [0, DEFAULT_PLOTLY_COLORS[idx % len(DEFAULT_PLOTLY_COLORS)]],
                        [1, DEFAULT_PLOTLY_COLORS[idx % len(DEFAULT_PLOTLY_COLORS)]],
                    ],
                    'showscale': False,
                })

            fig_annots = []

            if planes:
                basis_labels = [
                    f'σ<sub>1</sub>',
                    f'σ<sub>2</sub>',
                    f'σ<sub>3</sub>',
                ]
                for idx, plane in enumerate(planes):
                    plane_coords = maths_utils.get_plane_coords(
                        *plane,
                        side_length=(max_stress - min_stress) / 1.5,
                        resolution=10,
                    )
                    grid_corners = plane_coords['grid_corners']
                    plane_label = utils.format_axis_label(plane, basis_labels) + ' = 0'
                    fig_data.extend([
                        {
                            'type': 'mesh3d',
                            'x': grid_corners[0],
                            'y': grid_corners[1],
                            'z': grid_corners[2],
                            'i': [0, 2],
                            'j': [1, 3],
                            'k': [2, 0],
                            'showlegend': True,
                            'color': DEFAULT_PLOTLY_COLORS[
                                (len(yield_functions) + idx) % len(DEFAULT_PLOTLY_COLORS)
                            ],
                            'name': plane_label,
                        }
                    ])

            if show_axes:
                # TODO: `yld_func` is from the above loop:
                axes_dat, axes_annots = yld_func._get_3D_plot_axes(min_stress, max_stress)
                fig_data.extend(axes_dat)
                fig_annots.extend(axes_annots)

            if stress_states is not None:

                for idx, stresses in enumerate(stress_states):

                    if stresses is not None:
                        stresses = stresses.copy()

                        if normalise:
                            stresses /= eq_stress

                        fig_data.append(
                            {
                                'type': 'scatter3d',
                                'x': stresses[:, 0],
                                'y': stresses[:, 1],
                                'z': stresses[:, 2],
                                'name': f'{idx + 1}. Stress data',
                                'mode': 'markers',
                                'marker': {
                                    'color': DEFAULT_PLOTLY_COLORS[
                                        idx % len(DEFAULT_PLOTLY_COLORS)
                                    ],
                                }
                            }
                        )

                if join_stress_states:
                    max_idx = np.max(np.concatenate(stress_indices))
                    all_idx = [utils.expand_idx_array(i, max_idx) for i in stress_indices]

                    for i_idx, i in enumerate(all_idx[1:], 1):

                        if (stress_states[i_idx] is None or
                                stress_states[i_idx - 1] is None):
                            continue

                        A_exp, A_idx = i
                        B_exp, B_idx = all_idx[i_idx - 1]

                        matching_idx = np.ma.filled(A_exp == B_exp, fill_value=False)
                        stress_A_idx = A_idx.data[matching_idx]
                        stress_B_idx = B_idx.data[matching_idx]

                        A = stress_states[i_idx][stress_A_idx]
                        B = stress_states[i_idx - 1][stress_B_idx]

                        # Lines should go from A to B
                        xyz = np.zeros((3, A.shape[0] * 3))
                        xyz[:, 2::3] = np.nan
                        xyz[:, 0::3] = A.T
                        xyz[:, 1::3] = B.T

                        fig_data.append({
                            'type': 'scatter3d',
                            'x': xyz[0],
                            'y': xyz[1],
                            'z': xyz[2],
                            'mode': 'lines',
                            'name': f'Stress path ({i_idx})',
                            'line': {
                                'color': 'black',
                            },
                        })

            fig = graph_objects.FigureWidget(
                data=fig_data,
                layout={
                    'height': 600,
                    'scene': {
                        'aspectmode': 'data',
                        'camera': {
                            'projection': {
                                'type': 'orthographic',
                            },
                            'eye': {
                                'x': 2,
                                'z': 1,
                            }
                        },
                        'xaxis': {
                            'title': 'σ<sub>1</sub>',
                        },
                        'yaxis': {
                            'title': 'σ<sub>2</sub>',
                        },
                        'zaxis': {
                            'title': 'σ<sub>3</sub>',
                        },
                        'annotations': fig_annots,
                    },
                    **(layout or {}),
                },
            )

            return fig

    @classmethod
    def compare_2D(cls, yield_functions, plane, normalise=True, resolution=DEF_2D_RES,
                   equivalent_stress=None, min_stress=None, max_stress=None,
                   stress_states=None, up=None, show_contour_grid=False,
                   stress_indices=None, join_stress_states=False, legend_text=None,
                   show_numerical_lankford=False, show_numerical_lankford_fit=False,
                   layout=None, use_plotly_contour=False, sheet_dirs=None):
        """Visualise multiple yield functions in 2D.

        Parameters
        ----------
        join_stress_states : bool, optional
            If True, stress states corresponding to the same `stress_indices` values will
            be joined by straight lines. False by default.
        use_plotly_contour : bool, optional
            If True, plot each yield function as a Plotly contour plot with a single
            zero-contour. If False, calculate the zero-contour using scikit-image.

        """

        if stress_states is not None:
            if len(stress_states) != len(yield_functions):
                msg = ('If specified, `stress_states` should be a list of equal length '
                       'to the `yield_functions` list.')
                raise ValueError(msg)

            if join_stress_states and not stress_indices:
                msg = ('If `join_stress_states=True`, `stress_indices` must be specified '
                       'as a list (of equal length to `stress_states`) of integer '
                       'indices that determine how the stress states should be '
                       'connected.')
                raise ValueError(msg)

            if stress_indices and len(stress_indices) != len(stress_states):
                msg = '`stress_indices` must have the same length as `stress_states`.'
                raise ValueError(msg)

        if legend_text:
            if len(legend_text) != len(yield_functions):
                msg = ('If specified, `legend_text` must be a list of equal length to '
                       '`yield_function`.')
                raise ValueError(msg)

        stress_range = cls._get_multi_plot_stress_range(
            yield_functions=yield_functions,
            normalise=normalise,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
        )
        min_stress = stress_range['min_stress']
        max_stress = stress_range['max_stress']
        eq_stress = stress_range['eq_stress']

        grid_coords_2D, values_all, basis_unit = cls.get_2D_multi_plot_data(
            plane,
            yield_functions,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=eq_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            up=up
        )

        if normalise:
            unit = ' / σ<sub>eq. max.</sub>'
        else:
            grid_coords_2D /= 1e6
            unit = ' / MPa'

        fig_data = []
        for idx, (yld_func, i) in enumerate(zip(yield_functions, values_all)):

            name = f'{idx + 1}. {yld_func.__class__.__name__}'
            if legend_text:
                name += f' [{legend_text[idx]}] '

            yield_point = getattr(yld_func, 'yield_point', None)
            if yield_point:
                name += f' ({yield_point})'

            if show_numerical_lankford:

                num_lank = yld_func.get_numerical_lankford()
                name += f' [R = {num_lank["lankford"]:.3f}]'
                fit_centre = num_lank['fit_centre']

                if normalise:
                    grid_x = num_lank['grid_points'][0] - fit_centre
                    grid_x = (grid_x / eq_stress)
                    grid_x += (fit_centre / eq_stress)

                    grid_y = num_lank['grid_points'][1] / eq_stress
                else:
                    grid_x = num_lank['grid_points'][0] / 1e6
                    grid_y = num_lank['grid_points'][1] / 1e6

                if show_numerical_lankford_fit:
                    fig_data.append(
                        {
                            'type': 'scatter',
                            'x': grid_x[num_lank['grid_point_fitted_idx']],
                            'y': grid_y[num_lank['grid_point_fitted_idx']],
                            'mode': 'markers',
                            'marker': {
                                'size': 4,
                                'color': DEFAULT_PLOTLY_COLORS[
                                    idx % len(DEFAULT_PLOTLY_COLORS)
                                ],
                            },
                            'name': 'Tangent fit data',
                            'legendgroup': name,
                        }
                    )

                tangent_vec = num_lank['tangent_vector']
                tangent_mag = max_stress * 0.2
                t1 = - (0.5 * tangent_mag * tangent_vec)
                t2 = + (0.5 * tangent_mag * tangent_vec)
                tangent_line = np.vstack([t1, t2]).T

                normal_vec = num_lank['normal_vector']
                normal_vec *= tangent_mag * 1.3
                normal_line_seg = np.array([
                    [0, 0 + normal_vec[0]],
                    [0, normal_vec[1]]
                ])

                if normalise:
                    normal_line_seg[0] += (fit_centre / eq_stress)
                    tangent_line[0] += (fit_centre / eq_stress)
                else:
                    normal_line_seg = normal_line_seg / 1e6
                    normal_line_seg[0] += (fit_centre / 1e6)
                    tangent_line = tangent_line / 1e6
                    tangent_line[0] += (fit_centre / 1e6)

                fig_data.append(
                    {
                        'type': 'scatter',
                        'x': tangent_line[0],
                        'y': tangent_line[1],
                        'mode': 'lines',
                        'line': {
                            'color': DEFAULT_PLOTLY_COLORS[
                                idx % len(DEFAULT_PLOTLY_COLORS)
                            ],
                        },
                        'showlegend': False,
                        'name': name,
                        'legendgroup': name,
                    },
                )

                fig_data.append(
                    {
                        'type': 'scatter',
                        'x': normal_line_seg[0],
                        'y': normal_line_seg[1],
                        'mode': 'lines',
                        'line': {
                            'color': DEFAULT_PLOTLY_COLORS[
                                idx % len(DEFAULT_PLOTLY_COLORS)
                            ],
                        },
                        'showlegend': False,
                        'name': name,
                        'legendgroup': name,
                    },
                )

            if use_plotly_contour:
                fig_data.append({
                    'type': 'contour',
                    'x': grid_coords_2D[0],
                    'y': grid_coords_2D[1],
                    'z': i,
                    'contours': {
                        'start': 0,
                        'end': 0,
                        'size': 1,
                        'coloring': 'none',
                    },
                    'line': {
                        # Resolution must be high enough to accurately portray curvature;
                        # so don't rely on smoothing!
                        'smoothing': 0,
                        'color': DEFAULT_PLOTLY_COLORS[
                            idx % len(DEFAULT_PLOTLY_COLORS)
                        ],
                    },
                    'showscale': False,
                    'showlegend': True,
                    'hoverinfo': 'none',
                    'name': name,
                    'legendgroup': name,
                })
            else:
                yld_func_vals = i.reshape((resolution + 1, resolution + 1)).T
                contours_scaled = find_contours(yld_func_vals, 0)
                contours = utils.remap_contours(
                    grid_coords_2D=grid_coords_2D,
                    contours=contours_scaled,
                    resolution=resolution,
                )
                fig_data.append({
                    'type': 'scatter',
                    'x': contours[0][:, 0],
                    'y': contours[0][:, 1],
                    'line': {
                        'color': DEFAULT_PLOTLY_COLORS[
                            idx % len(DEFAULT_PLOTLY_COLORS)
                        ],
                        'width': 0.5,
                    },
                    'showlegend': True,
                    'hoverinfo': 'none',
                    'name': name,
                    'legendgroup': name,
                })

        if stress_states is not None:

            proj_stresses = []  # 1 element for each yield_function/stress_state
            for stresses in stress_states:
                if stresses is not None:
                    stresses = stresses.copy()
                    if normalise:
                        stresses /= eq_stress

                    # TODO: Order stress states by distance to viewer.

                    # Colour/set opacity for stress states by distance to projection plane:
                    # stresses shape (N, 3)
                    normal = np.array(plane)
                    normal_unit = normal / np.linalg.norm(normal)
                    dist = np.dot(stresses, normal_unit)
                    proj = stresses
                    proj -= (normal_unit * np.dot(stresses, normal)[:, None])

                    # Rotate projected states so plane normal is aligned with z-axis
                    basis_full = np.hstack([basis_unit, normal_unit[:, None]])
                    proj_rot = basis_full.T @ proj.T

                    if not normalise:
                        proj_rot[0:2] /= 1e6

                    proj_stresses.append({
                        'x': proj_rot[0],
                        'y': proj_rot[1],
                        'dist': dist
                    })
                else:
                    proj_stresses.append(None)

            for idx, proj in enumerate(proj_stresses):
                if proj is not None:
                    color_dim = np.abs(proj['dist']) / np.max(np.abs(proj['dist']))
                    base_col = DEFAULT_PLOTLY_COLORS[idx % len(DEFAULT_PLOTLY_COLORS)]
                    base_col_rgb = utils.parse_rgb_str(base_col)

                    fig_data.append({
                        'type': 'scatter',
                        'x': proj['x'],
                        'y': proj['y'],
                        'mode': 'markers',
                        'name': f'{idx+ 1}. Stress data',
                        'hovertext': dist,
                        'marker': {
                            'color': color_dim,
                            'showscale': False,
                            'colorscale': [
                                [0,   utils.format_rgba(*base_col_rgb, 1.0)],
                                [0.2, utils.format_rgba(*base_col_rgb, 0.2)],
                                [1,   utils.format_rgba(*base_col_rgb, 0.0)],
                            ],
                        },
                    })

            if join_stress_states:
                max_idx = np.max(np.concatenate(stress_indices))
                all_idx = [utils.expand_idx_array(i, max_idx) for i in stress_indices]

                for i_idx, i in enumerate(all_idx[1:], 1):

                    if proj_stresses[i_idx] is None or proj_stresses[i_idx - 1] is None:
                        continue

                    A_exp, A_idx = i
                    B_exp, B_idx = all_idx[i_idx - 1]

                    matching_idx = np.ma.filled(A_exp == B_exp, fill_value=False)
                    stress_A_idx = A_idx.data[matching_idx]
                    stress_B_idx = B_idx.data[matching_idx]

                    A_x = proj_stresses[i_idx]['x'][stress_A_idx]
                    A_y = proj_stresses[i_idx]['y'][stress_A_idx]
                    B_x = proj_stresses[i_idx - 1]['x'][stress_B_idx]
                    B_y = proj_stresses[i_idx - 1]['y'][stress_B_idx]

                    # Lines should go from A to B
                    xy = np.zeros((2, A_x.size * 3))
                    xy[:, 2::3] = np.nan
                    xy[0, 0::3] = A_x
                    xy[0, 1::3] = B_x
                    xy[1, 0::3] = A_y
                    xy[1, 1::3] = B_y

                    # Plotly doesn't support varying line color, but if it did, we would
                    # do this:
                    avg_dist = 0.5 * (
                        proj_stresses[i_idx]['dist'][stress_A_idx] +
                        proj_stresses[i_idx - 1]['dist'][stress_B_idx]
                    )
                    color_dim = np.abs(avg_dist) / np.max(np.abs(avg_dist))

                    fig_data.append({
                        'type': 'scatter',
                        'x': xy[0],
                        'y': xy[1],
                        'mode': 'lines',
                        'name': f'Stress path ({i_idx})',
                        'line': {
                            'width': 0.4,
                            'color': 'rgba(120, 120, 120, 0.5)',
                        },
                    })

        if show_contour_grid:

            cg_text_strs = []
            for idx, (yld_func, i) in enumerate(zip(yield_functions, values_all)):
                cg_text_i = f'{idx + 1}. {yld_func.__class__.__name__}'
                if legend_text:
                    cg_text_i += f' [{legend_text[idx]}] '
                cg_text_strs.append(cg_text_i)

            cg_text_all = []
            for cg_text_i, i in zip(cg_text_strs, values_all):
                cg_text_all.append([f'{cg_text_i}: {j:.3f}' for j in i])

            cg_text_final = ['<br>'.join(list(i)) for i in zip(*cg_text_all)]

            fig_data.append(
                {
                    'type': 'scatter',
                    'x': grid_coords_2D[0],
                    'y': grid_coords_2D[1],
                    'mode': 'markers',
                    'text': cg_text_final,
                    'marker': {
                        'size': 3,
                        'color': 'gray',
                        'opacity': 0.5,
                    },
                    'name': 'Contour grid',
                },
            )

        basis_labels = [
            f'σ<sub>11</sub>',
            f'σ<sub>22</sub>',
            f'σ<sub>33</sub>',
        ]
        basis_labels_unit = [f'{i}{unit}' for i in basis_labels]
        axis_labels = {
            'x': utils.format_axis_label(
                basis_unit[:, 0],
                basis_labels_unit,
                sheet_dirs=sheet_dirs,
            ),
            'y': utils.format_axis_label(
                basis_unit[:, 1],
                basis_labels_unit,
                sheet_dirs=sheet_dirs,
            ),
        }
        fig_annots = []
        annot_font = {
            'size': 14,
        }
        if equivalent_stress:
            fig_eq_stress = [equivalent_stress]
        else:
            fig_eq_stress = [getattr(i, 'equivalent_stress', None)
                             for i in yield_functions]

        if any(fig_eq_stress):
            fig_eq_stress = [i / 1e6 for i in fig_eq_stress if i]
            fig_eq_stress_list = []
            for i in fig_eq_stress:
                if i:
                    fig_eq_stress_list.append(f'{i:.0f} MPa')
                else:
                    fig_eq_stress_list.append('n/a')
            fig_eq_stress_fmt = ', '.join(fig_eq_stress_list)
            if len(fig_eq_stress_list) > 1:
                fig_eq_stress_fmt = f'[{fig_eq_stress_fmt}]'

            eq_stress_annot = {
                'text': f'Equivalent stress, σ<sub>eq.</sub> = {fig_eq_stress_fmt}',
                'font': annot_font,
                'showarrow': False,
                'x': 0.04,
                'y': 0.98,
                'xanchor': 'left',
                'yanchor': 'top',
                'xref': 'paper',
                'yref': 'paper',
            }
            fig_annots.append(eq_stress_annot)

        fig_annots.append({
            'text': utils.format_axis_label(plane, basis_labels) + ' = 0',
            'font': annot_font,
            'showarrow': False,
            'x': 0.04,
            'y': 0.88,
            'xanchor': 'left',
            'yanchor': 'top',
            'xref': 'paper',
            'yref': 'paper',
        })

        fig = graph_objects.FigureWidget(
            data=fig_data,
            layout={
                'height': 600,
                'width': 600,
                'xaxis': {
                    'scaleanchor': 'y',
                    'title': axis_labels['x'],
                },
                'yaxis': {
                    'title': axis_labels['y'],
                },
                'annotations': fig_annots,
                'legend': {
                    'yanchor': 'top',
                    'xanchor': 'center',
                    'y': -0.15,
                    'x': 0.5,
                    'tracegroupgap': 5,
                },
                **(layout or {}),
            }
        )

        return fig

    @classmethod
    def _get_multi_plot_stress_range(cls, yield_functions, normalise,
                                     equivalent_stress=None, min_stress=None,
                                     max_stress=None):
        """Set a minimum/maximum stress and equivalent stress for plotting.

        Parameters
        ----------
        yield_functions : list of YieldFunction
        normalise : bool
        equivalent_stress : number
        min_stress : number
        max_stress : number

        Notes
        -----
        If `normalise` is True, an equivalent stress is used to normalise the yield
        function and any superimposed stress states. If none of the yield function
        definitions include an `equivalent_stress` attribute, then `equivalent_stress`
        must be passed to this method. If multiple yield function definitions include
        an `equivalent_stress` attribute, then the largest value is used.

        If `normalise` is False, the return `eq_stress` is set to one. If the yield
        function definitions includes an `equivalent_stress` attribute, then the min/max
        stress values are set as factors of the largest `equivalent_stress` attribute,
        otherwise, min/max stress values must be passed to this method.

        """

        all_eq_stresses = [
            getattr(i, 'equivalent_stress')
            for i in yield_functions
            if hasattr(i, 'equivalent_stress')
        ]

        if normalise:
            eq_stress = None
            if all_eq_stresses:
                # Normalise with respect to the largest equivalent stress:
                eq_stress = max(all_eq_stresses)

            if eq_stress is None:
                if equivalent_stress is not None:
                    eq_stress = equivalent_stress
                else:
                    msg = ('If `normalise=True`, you must specify an `equivalent_stress`,'
                           ' since none of the yield functions have an '
                           '`equivalent stress` attribute.')
                    raise ValueError(msg)

            minmax_stress = 2
            if min_stress is None:
                min_stress = -minmax_stress
            if max_stress is None:
                max_stress = +minmax_stress

        else:
            eq_stress = 1
            if equivalent_stress is not None:
                msg = ('No need to specify `equivalent_stress` unless you also specify '
                       '`normalise`')
                raise ValueError(msg)

            if all_eq_stresses:
                minmax_stress = max(all_eq_stresses)
                min_stress = -2 * minmax_stress
                max_stress = +2 * minmax_stress

            else:
                if min_stress is None or max_stress is None:
                    msg = ('Specify `min_stress` and `max_stress` to determine the 3D '
                           'grid on which the yield surfaces are shown.')
                    raise ValueError(msg)

        out = {
            'min_stress': min_stress,
            'max_stress': max_stress,
            'eq_stress': eq_stress,
        }
        return out

    def _get_3D_plot_axes(self, min_stress, max_stress):

        axis_lines = {
            'mode': 'lines',
            'line': {
                'color': 'gray',
            },
            'name': 'Axes',
            'legendgroup': 'Axes',
            'showlegend': False,
            'hoverinfo': 'none',
            'projection': {
                'x': {
                    'show': False,
                },
                'y': {
                    'show': False,
                },
                'z': {
                    'show': False,
                },
            }
        }

        data = [
            {
                'type': 'scatter3d',
                'x': [min_stress, max_stress],
                'y': [0, 0],
                'z': [0, 0],
                **axis_lines,
            },
            {
                'type': 'scatter3d',
                'x': [0, 0],
                'y': [min_stress, max_stress],
                'z': [0, 0],
                **axis_lines,
            },
            {
                'type': 'scatter3d',
                'x': [0, 0],
                'y': [0, 0],
                'z': [min_stress, max_stress],
                **axis_lines,
            },
        ]

        label_dist = max_stress * 1.1

        annots = [
            {
                'showarrow': False,
                'x': label_dist,
                'y': 0,
                'z': 0,
                'text': 'σ<sub>1</sub>',
                'font': {
                    'size': 18,
                }
            },
            {
                'showarrow': False,
                'x': 0,
                'y': label_dist,
                'z': 0,
                'text': 'σ<sub>2</sub>',
                'font': {
                    'size': 18,
                }
            },
            {
                'showarrow': False,
                'x': 0,
                'y': 0,
                'z': label_dist,
                'text': 'σ<sub>3</sub>',
                'font': {
                    'size': 18,
                }
            },
        ]

        return data, annots

    @classmethod
    def get_3D_multi_plot_data(cls, yield_functions, normalise=True, resolution=DEF_3D_RES,
                               equivalent_stress=None, min_stress=None, max_stress=None):

        if (
            equivalent_stress is None or
            min_stress is None or
            max_stress is None
        ):
            stress_range = cls._get_multi_plot_stress_range(
                yield_functions,
                normalise,
                equivalent_stress=equivalent_stress,
                min_stress=min_stress,
                max_stress=max_stress,
            )
            min_stress = stress_range['min_stress']
            max_stress = stress_range['max_stress']
            equivalent_stress = stress_range['eq_stress']

        (stress_X, stress_Y, stress_Z), stress_grid = cls._get_3D_plot_contour_grid(
            min_stress,
            max_stress,
            resolution=resolution,
        )

        grid_values_all = []
        for yld_func in yield_functions:
            values_i = yld_func.get_value(stress_grid * equivalent_stress)
            # Normalise values solely for the purpose of the visualisation (so contour
            # values are of the order 1):
            abs_max = np.nanmax(np.abs(values_i))
            values_i /= abs_max
            grid_values_all.append(values_i)

        return (stress_X, stress_Y, stress_Z), grid_values_all

    @classmethod
    def get_2D_multi_plot_data(cls, plane, yield_functions, normalise=True,
                               resolution=DEF_2D_RES, equivalent_stress=None,
                               min_stress=None, max_stress=None, up=None):
        if (
            equivalent_stress is None or
            min_stress is None or
            max_stress is None
        ):
            stress_range = cls._get_multi_plot_stress_range(
                yield_functions,
                normalise,
                equivalent_stress=equivalent_stress,
                min_stress=min_stress,
                max_stress=max_stress,
            )
            min_stress = stress_range['min_stress']
            max_stress = stress_range['max_stress']
            equivalent_stress = stress_range['eq_stress']

        plane_coords = maths_utils.get_plane_coords(
            *plane,
            side_length=(max_stress - min_stress) / 1.5,
            resolution=resolution,
            up=up,
        )
        grid_coords_2D = plane_coords['grid_coords_2D']
        grid_coords = plane_coords['grid_coords']
        basis_unit = plane_coords['basis_unit']

        stress_grid = np.zeros(((resolution + 1)**2, 3, 3))
        stress_grid[:, 0, 0] = grid_coords[0]
        stress_grid[:, 1, 1] = grid_coords[1]
        stress_grid[:, 2, 2] = grid_coords[2]

        grid_values_all = []
        for yld_func in yield_functions:
            values_i = yld_func.get_value(stress_grid * equivalent_stress)
            # Normalise values solely for the purpose of the visualisation (so contour
            # values are of the order 1):
            abs_max = np.nanmax(np.abs(values_i))
            values_i /= abs_max
            grid_values_all.append(values_i)

        return grid_coords_2D, grid_values_all, basis_unit

    @staticmethod
    def _get_3D_plot_contour_grid(min_stress, max_stress, resolution=DEF_3D_RES):

        stress_vals = np.linspace(min_stress, max_stress, num=resolution)
        stress_X, stress_Y, stress_Z = np.meshgrid(stress_vals, stress_vals, stress_vals)

        stress_grid = np.zeros((resolution**3, 3, 3))
        stress_grid[:, 0, 0] = stress_X.flatten()
        stress_grid[:, 1, 1] = stress_Y.flatten()
        stress_grid[:, 2, 2] = stress_Z.flatten()

        # Remove the zero stress point:
        zero_idx = np.all(stress_grid == np.zeros((3, 3)), axis=(-2, -1))
        non_zero_idx = np.logical_not(zero_idx)
        stress_grid = stress_grid[non_zero_idx]

        return (stress_X, stress_Y, stress_Z), stress_grid

    def get_3D_plot_data(self, normalise=True, resolution=DEF_3D_RES,
                         equivalent_stress=None, min_stress=None, max_stress=None):

        (stress_X, stress_Y, stress_Z), grid_values_all = self.get_3D_multi_plot_data(
            [self],
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
        )

        grid_values = grid_values_all[0]

        return (stress_X, stress_Y, stress_Z), grid_values

    def get_2D_plot_data(self, plane, normalise=True, resolution=DEF_2D_RES,
                         equivalent_stress=None, min_stress=None, max_stress=None,
                         up=None):

        grid_coords_2D, grid_values_all, basis_unit = self.get_2D_multi_plot_data(
            plane,
            [self],
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            up=up,
        )

        grid_values = grid_values_all[0]

        return grid_coords_2D, grid_values, basis_unit

    def show_3D(self, normalise=True, resolution=DEF_3D_RES, equivalent_stress=None,
                min_stress=None, max_stress=None, show_axes=True, planes=None,
                stress_states=None, backend='plotly', stress_indices=None,
                join_stress_states=False, show_contour_grid=False, legend_text=None,
                layout=None):
        """
        Parameters
        ----------
        stress_states : ndarray of shape (N, 3)
            Principle stress states to add to the visualisation.

        """
        return self.compare_3D(
            yield_functions=[self],
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            show_axes=show_axes,
            planes=planes,
            stress_states=stress_states,
            backend=backend,
            stress_indices=stress_indices,
            join_stress_states=join_stress_states,
            show_contour_grid=show_contour_grid,
            legend_text=legend_text,
            layout=layout,
        )

    def show_2D(self, plane, normalise=True, resolution=DEF_2D_RES,
                equivalent_stress=None, min_stress=None, max_stress=None,
                stress_states=None, up=None, show_contour_grid=False, stress_indices=None,
                join_stress_states=False, legend_text=None,
                show_numerical_lankford=False, show_numerical_lankford_fit=False,
                layout=None, use_plotly_contour=False):
        """
        Parameters
        ----------
        stress_states : ndarray of shape (N, 3)
            Principle stress states to add to the visualisation.

        """
        return self.compare_2D(
            [self],
            plane,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            stress_states=stress_states,
            up=up,
            show_contour_grid=show_contour_grid,
            stress_indices=stress_indices,
            join_stress_states=join_stress_states,
            legend_text=legend_text,
            show_numerical_lankford=show_numerical_lankford,
            show_numerical_lankford_fit=show_numerical_lankford_fit,
            layout=layout,
            use_plotly_contour=use_plotly_contour,
        )

    def get_numerical_lankford(self, fit_domain=0.05, resolution=300, tol=1e-4):
        """Get the Lankford coefficient by numerically fitting the tangent to the yield
        function.

        Parameters
        ----------
        fit_domain : float, optional
            The side lengths of the box within which to fit a line (i.e. the yield
            function tangent), expressed as a fraction of the equivalent stress.
        resolution : int, optional
            Size of the grid within the fitting domain, on which values of the yield
            function will be evaluated.
        tol : float, optional
            Yield function values that are within this value from zero will be included
            in the tangent fit.

        Returns
        -------
        lankford : float

        """

        if not hasattr(self, 'equivalent_stress'):
            msg = 'Yield function must have an `equivalent_stress` parameter.'
            raise NotImplementedError(msg)
        else:
            equiv_stress = getattr(self, 'equivalent_stress')

        # Identify a uniaxial stress state that is close to the yield surface (it will
        # be close to the "equivalent stress", but for some reason they don't always
        # precisely coincide):
        fit_centre_range = np.linspace(-0.5, 0.5, num=10000) * equiv_stress
        fit_centre_range += equiv_stress
        num_ss = fit_centre_range.shape[0]
        stress_states = np.tile(np.eye(3), num_ss).reshape(num_ss, 3, 3)
        stress_states[:, 0, 0] = fit_centre_range
        fit_centre_vals = self.get_value(stress_states)
        fit_centre_idx = np.argmin(np.abs(fit_centre_vals))
        fit_centre = fit_centre_range[fit_centre_idx]

        X, Y = np.meshgrid(*[np.linspace(-0.5, 0.5, num=resolution)] * 2)
        xy = np.vstack([X.flatten(), Y.flatten()])
        xy *= fit_centre * fit_domain
        xy += np.array([[fit_centre, 0]]).T

        num_ss = xy.shape[1]
        stress_states = np.tile(np.eye(3), (num_ss, 1)).reshape(num_ss, 3, 3)
        stress_states[:, 0, 0] = xy[0]
        stress_states[:, 1, 1] = xy[1]

        vals = self.get_value(stress_states)
        good_idx = np.where((np.abs(vals) - tol) < 0)[0]
        if not good_idx.size:
            raise ValueError(f'No yield surface solutions within `tol={tol}`.')

        fit_coords = xy[:, good_idx]
        popt, pcov = curve_fit(maths_utils.line, fit_coords[0], fit_coords[1])
        m = popt[0]

        tangent_vector = np.array([1, m])
        tangent_vector_unit = tangent_vector / np.linalg.norm(tangent_vector)

        normal = -1 * np.array([-m, 1])  # -1 to point out from the surface
        normal_n = normal / np.linalg.norm(normal)

        lankford = normal_n[1] / -(normal_n[0] + normal_n[1])

        out = {
            'grid_points': xy,
            'grid_point_fitted_idx': good_idx,
            'normal_vector': normal_n,
            'tangent_vector': tangent_vector_unit,
            'lankford': lankford,
            'fit_centre': fit_centre,
        }

        return out


class YieldFunctionEvolutionViz:
    """Class to hold multiple LoadResponseSet objects and generate a widget
    for examining the time-evolution of fitted yield functions."""

    def __init__(self, load_response_sets, plane, yield_function_names=None,
                 sheet_dirs=None, normalise=True, resolution=100, equivalent_stress=None,
                 min_stress=None, max_stress=None, up=None):
        """
        Parameters
        ----------
        load_response_sets : list of LoadResponseSet
            List of LoadResponseSet objects whose fitted yield functions are to be
            included in the animation. Fitted yield functions from all objects will be
            collected and animated according to their respective yield point criteria.
        yield_function_names : list of str, optional
            If specified, only these yield function names will be included as separate
            traces.    

        """

        self.load_response_sets = load_response_sets
        self.yield_function_names = yield_function_names
        self.plane = plane
        self.normalise = normalise
        self.sheet_dirs = sheet_dirs

        # keyed by yield function name:
        fitted_yld_funcs_by_name = {}
        yield_points_by_name = {}

        self.YPC_name = None
        for load_resp_set in self.load_response_sets:

            for i in load_resp_set.fitted_yield_functions:

                yld_point = load_resp_set.yield_point_criteria[
                    i['YPC_idx']
                ].values[0, i['YPC_value_idx']]

                if self.YPC_name is None:
                    self.YPC_name = load_resp_set.yield_point_criteria[i['YPC_idx']].source
                elif load_resp_set.yield_point_criteria[i['YPC_idx']].source != self.YPC_name:
                    msg = 'Different yield point criteria sources used.'
                    raise NotImplementedError(msg)

                yld_func_name = i['yield_function'].name

                if yld_func_name not in fitted_yld_funcs_by_name:
                    fitted_yld_funcs_by_name.update({yld_func_name: []})
                if yld_func_name not in yield_points_by_name:
                    yield_points_by_name.update({yld_func_name: []})

                yld_point_str = f'{yld_point:.10f}'

                if yld_point_str in yield_points_by_name:
                    msg = (f'Yield point "{yld_point_str}" for yield function '
                           f'"{yld_func_name}" appears more than once!')
                    raise NotImplementedError(msg)
                else:
                    fitted_yld_funcs_by_name[yld_func_name].append(i['yield_function'])
                    yield_points_by_name[yld_func_name].append(yld_point_str)

        all_yield_points = [
            f'{i:.10f}' for i in
            sorted(set([
                float(j)
                for i in yield_points_by_name.values()
                for j in i
            ]))
        ]

        self.all_yield_points = all_yield_points
        self.fitted_yld_funcs_by_name = fitted_yld_funcs_by_name
        self.yield_points_by_name = yield_points_by_name
        self.yield_func_order = list(self.fitted_yld_funcs_by_name.keys())
        self.all_yield_functions = [
            j
            for i in self.yield_func_order
            for j in self.fitted_yld_funcs_by_name[i]
        ]

        self.contours_by_name = self._get_contours(
            resolution,
            normalise,
            equivalent_stress,
            min_stress,
            max_stress,
            up,
        )

        self._widgets = self._generate_widgets()
        self._visual = self._generate_visual()

    @property
    def num_yield_points(self):
        return len(self.all_yield_points)

    def _get_contours(self, resolution, normalise, equivalent_stress, min_stress,
                      max_stress, up):

        stress_range = YieldFunction._get_multi_plot_stress_range(
            yield_functions=self.all_yield_functions,
            normalise=normalise,
            equivalent_stress=equivalent_stress,
            min_stress=min_stress,
            max_stress=max_stress,
        )
        min_stress = stress_range['min_stress']
        max_stress = stress_range['max_stress']
        eq_stress = stress_range['eq_stress']

        grid_coords_2D, values_all, basis_unit = YieldFunction.get_2D_multi_plot_data(
            self.plane,
            self.all_yield_functions,
            normalise=normalise,
            resolution=resolution,
            equivalent_stress=eq_stress,
            min_stress=min_stress,
            max_stress=max_stress,
            up=up
        )

        self.basis_unit = basis_unit
        if not normalise:
            grid_coords_2D /= 1e6

        contours_by_name = {}
        for yld_func, i in zip(self.all_yield_functions, values_all):
            yld_func_vals = i.reshape((resolution + 1, resolution + 1)).T
            contours_scaled = find_contours(yld_func_vals, 0)
            contours = utils.remap_contours(
                grid_coords_2D=grid_coords_2D,
                contours=contours_scaled,
                resolution=resolution,
            )

            if yld_func.name not in contours_by_name:
                contours_by_name.update({yld_func.name: []})
            contours_by_name[yld_func.name].append(contours)

        all_contour_coords = np.concatenate([
            i[0] for v in contours_by_name.values() for i in v
        ])
        maxs = np.max(all_contour_coords, axis=0)
        mins = np.min(all_contour_coords, axis=0)
        max_pad = [i * 0.1 for i in maxs]
        min_pad = [i * 0.1 for i in mins]
        self.x_range = [mins[0] + min_pad[0], maxs[0] + max_pad[0]]
        self.y_range = [mins[1] + min_pad[1], maxs[1] + max_pad[1]]

        return contours_by_name

    def _generate_widgets(self):
        """Generate the widgets for the visualisation."""

        contours_first = [
            self.contours_by_name[name][
                self.yield_points_by_name[name].index(self.all_yield_points[0])
            ]
            for name in self.yield_func_order
        ]

        fig_data = []
        for idx, c in enumerate(contours_first):
            name = f'{self.yield_func_order[idx]}'
            fig_data.append({
                'x': c[0][:, 0],
                'y': c[0][:, 1],
                'name': name,
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[
                        idx % len(DEFAULT_PLOTLY_COLORS)
                    ],
                    'width': 0.5,
                },
                'showlegend': True,
                'hoverinfo': 'none',
            })
        annot_font = {
            'size': 14,
        }

        if self.normalise:
            unit = ' / σ<sub>eq. max.</sub>'
        else:
            unit = ' / MPa'

        basis_labels = [
            f'σ<sub>11</sub>',
            f'σ<sub>22</sub>',
            f'σ<sub>33</sub>',
        ]
        basis_labels_unit = [f'{i}{unit}' for i in basis_labels]
        axis_labels = {
            'x': utils.format_axis_label(
                self.basis_unit[:, 0],
                basis_labels_unit,
                sheet_dirs=self.sheet_dirs
            ),
            'y': utils.format_axis_label(
                self.basis_unit[:, 1],
                basis_labels_unit,
                sheet_dirs=self.sheet_dirs
            ),
        }

        fig_wig = graph_objects.FigureWidget(
            data=fig_data,
            layout={
                'height': 600,
                'width': 600,
                'xaxis': {
                    'scaleanchor': 'y',
                    'range': self.x_range,
                    'title': axis_labels['x'],
                },
                'yaxis': {
                    'range': self.y_range,
                    'title': axis_labels['y'],
                },
                'legend': {
                    'yanchor': 'top',
                    'xanchor': 'center',
                    'y': -0.15,
                    'x': 0.5,
                    'tracegroupgap': 5,
                },
                'annotations': [
                    {
                        'text': f'{self.YPC_name} = {float(self.all_yield_points[0]):.6f}',
                        'font': annot_font,
                        'showarrow': False,
                        'x': 0.04,
                        'y': 0.98,
                        'xanchor': 'left',
                        'yanchor': 'top',
                        'xref': 'paper',
                        'yref': 'paper',
                    }
                ]
            }
        )

        widget_dict = {
            'figure': fig_wig,
            'play': widgets.Play(
                value=0,
                min=0,
                max=self.num_yield_points - 1,
                step=1,
                interval=50,
                description="Press play",
                disabled=False,
            ),
            'slider': widgets.IntSlider(
                value=0,
                min=0,
                max=self.num_yield_points - 1,
                step=1,
                readout=False,
            ),
        }
        widgets.jslink((widget_dict['play'], 'value'), (widget_dict['slider'], 'value'))
        widget_dict['slider'].observe(self._update_figure, names='value')
        return widget_dict

    def _update_figure(self, change):

        yld_point_idx = self._widgets['slider'].value
        yld_point = self.all_yield_points[yld_point_idx]
        fig = self._widgets['figure']
        with fig.batch_update():
            # loop over distinct yield functions and update their trace data to the
            # specified yield point:
            for trace_idx, name in enumerate(self.yield_func_order):

                yld_func_idx = self.yield_points_by_name[name].index(yld_point)
                try:
                    contour = self.contours_by_name[name][yld_func_idx]
                except KeyError:
                    contour = None

                if contour is not None:
                    fig.data[trace_idx].x = contour[0][:, 0]
                    fig.data[trace_idx].y = contour[0][:, 1]
                else:
                    fig.data[trace_idx].x = []
                    fig.data[trace_idx].y = []

                fig.layout.annotations[0].update({
                    'text': f'{self.YPC_name} = {float(yld_point):.6f}',
                })

    def _generate_visual(self):
        out = widgets.VBox(
            children=[
                self._widgets['figure'],
                widgets.HBox(
                    children=[self._widgets['play'], self._widgets['slider']]
                ),
            ]
        )
        return out

    @property
    def visual(self):
        return self._visual


def animate_yield_function_evolution(load_response_sets, plane, yield_function_names=None,
                                     sheet_dirs=None, _return_obj=False, **kwargs):
    """Generate a widget visualisation of the evolution of a set of yield functions,
    for viewing within a Jupyter notebook environment.

    Parameters
    ----------
    load_response_sets : list of LoadResponseSet
        List of LoadResponseSet objects whose fitted yield functions are to be included in
        the animation. Fitted yield functions from all objects will be collected and 
        animated according to their respective yield point criteria.
    yield_function_names : list of str, optional
        If specified, only these yield function names will be included as separate traces.    

    Returns
    -------
    ipwidgets.widgets.Box
        A widget containing a Plotly FigureWidget and control widgets for evolving the 
        yield point.

    """

    YFEV = YieldFunctionEvolutionViz(
        load_response_sets,
        plane,
        yield_function_names,
        sheet_dirs,
        **kwargs
    )
    if _return_obj:
        return YFEV
    else:
        return YFEV.visual
