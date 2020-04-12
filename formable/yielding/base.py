"""`formable.yielding.base.py`

Contains an abstract base class to represent a yield function. All yield function classes
inherit from this class.

"""

import abc
import functools

import numpy as np
import pyvista as pv
from plotly import graph_objects
from plotly.colors import DEFAULT_PLOTLY_COLORS
from scipy.optimize import least_squares

from formable import utils, maths_utils

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
    def from_fit(cls, stress_states, initial_params=None, **kwargs):
        """Fit the yield function to yield stress states.

        Parameters
        ----------
        stress_states : ndarray of shape (N, 3, 3)
        initial_params : dict 
            Any initial guesses for the fitting parameters. Mutually exclusive with
            additional keyword arguments passed, which are considered to be fixed.
        kwargs : dict
            Additional parameters passed to this method will be fixed during the fit.

        Returns
        -------
        yield_function : YieldFunction
            The fitted yield function.

        """

        fitting_param_names = [i for i in cls.PARAMETERS if i not in kwargs]
        initial_params_all = np.ones(len(fitting_param_names))

        initial_params = initial_params or {}

        if initial_params:
            for k, initial_guess in initial_params.items():
                if k not in fitting_param_names:
                    msg = (f'Initial guess specified for parameter "{k}", but this '
                           'parameter has also been specified as a keyword argument, '
                           'indicating it should be fixed.')
                    raise ValueError(msg)
                param_idx = fitting_param_names.index(k)
                initial_params_all[param_idx] = initial_guess

        if not fitting_param_names:

            # Construct yield function, but no need to fit:
            yield_function = cls(**kwargs)
            fit = None

        else:

            if len(fitting_param_names) > len(stress_states):
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

            fit = least_squares(
                cls.residual,
                initial_params_all,
                kwargs=dict(
                    stress_states=stress_states,
                    fitting_param_names=fitting_param_names,
                    **kwargs,
                )
            )

            parameters = dict(zip(fitting_param_names, fit.x))
            parameters.update(**kwargs)
            yield_function = cls(**parameters)

        yield_function.fit = fit

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

    @classmethod
    def compare_3D(cls, yield_functions, normalise=True, resolution=DEF_3D_RES,
                   equivalent_stress=None, min_stress=None, max_stress=None,
                   show_axes=True, planes=None, stress_states=None, backend='plotly'):
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

            grid = pv.StructuredGrid(stress_X, stress_Y, stress_Z)
            for yld_func, i in zip(yield_functions, values_all):
                grid[yld_func.__class__.__name__] = i

            # TODO: fix this, need a contours object for each yield surface?
            contours = grid.contour([0])

            pv.set_plot_theme('document')
            p = pv.PlotterITK()
            p.add_mesh(contours, scalars=contours.points[:, 2])
            if stress_states is not None:
                p.add_points(stress_states)

            return p

        elif backend == 'plotly':

            stress_X = stress_X.flatten()
            stress_Y = stress_Y.flatten()
            stress_Z = stress_Z.flatten()

            fig_data = []

            for idx, (yld_func, i) in enumerate(zip(yield_functions, values_all)):

                name = f'{idx + 1}. {yld_func.__class__.__name__}'
                yield_point = getattr(yld_func, 'yield_point', None)
                if yield_point:
                    name += f' ({yield_point})'

                fig_data.append({
                    'type': 'isosurface',
                    'x': stress_X,
                    'y': stress_Y,
                    'z': stress_Z,
                    'value': i,
                    'isomin': 0,
                    'isomax': 0,
                    'surface_count': 1,
                    'opacity': 0.5,
                    'colorscale': [
                        [0, DEFAULT_PLOTLY_COLORS[idx]],
                        [1, DEFAULT_PLOTLY_COLORS[idx]],
                    ],
                    'showscale': False,
                    'showlegend': True,
                    'name': name,
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
                            'color': DEFAULT_PLOTLY_COLORS[len(yield_functions) + idx],
                            'name': plane_label,
                        }
                    ])

            if show_axes:
                # TODO: `yld_func` is from the above loop:
                axes_dat, axes_annots = yld_func._get_3D_plot_axes(min_stress, max_stress)
                fig_data.extend(axes_dat)
                fig_annots.extend(axes_annots)

            if stress_states is not None:

                for idx, (yld_func, stresses) in enumerate(zip(yield_functions, stress_states)):

                    if stresses is not None:

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
                                    'color': DEFAULT_PLOTLY_COLORS[idx],
                                }
                            }
                        )

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
                },
            )

            return fig

    @classmethod
    def compare_2D(cls, yield_functions, plane, normalise=True, resolution=DEF_2D_RES,
                   equivalent_stress=None, min_stress=None, max_stress=None,
                   stress_states=None, up=None, show_contour_grid=False):
        'Visualise multiple yield functions in 2D.'

        if stress_states is not None:
            if len(stress_states) != len(yield_functions):
                msg = ('If specified, `stress_states` should be a list of equal length '
                       'to the `yield_functions` list.')
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
            unit = ' / σ<sub>0</sub>'
        else:
            grid_coords_2D /= 1e6
            unit = ' / MPa'

        fig_data = []
        for idx, (yld_func, i) in enumerate(zip(yield_functions, values_all)):

            name = f'{idx + 1}. {yld_func.__class__.__name__}'
            yield_point = getattr(yld_func, 'yield_point', None)
            if yield_point:
                name += f' ({yield_point})'

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
                    # Resolution must be high enough to accurately portray curvature.
                    'smoothing': 0,
                    'color': DEFAULT_PLOTLY_COLORS[idx],
                },
                'showscale': False,
                'showlegend': True,
                'hoverinfo': 'none',
                'name': name,
            })

        if stress_states is not None:

            for idx, (yld_func, stresses) in enumerate(zip(yield_functions, stress_states)):

                if stresses is not None:

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

                    color_dim = np.abs(dist) / np.max(np.abs(dist))

                    base_col = DEFAULT_PLOTLY_COLORS[idx]
                    base_col_rgb = utils.parse_rgb_str(base_col)

                    fig_data.append({
                        'type': 'scatter',
                        'x': proj_rot[0],
                        'y': proj_rot[1],
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

        if show_contour_grid:
            fig_data.append(
                {
                    'type': 'scatter',
                    'x': grid_coords_2D[0],
                    'y': grid_coords_2D[1],
                    'mode': 'markers',
                    'marker': {
                        'size': 3,
                        'color': 'gray',
                        'opacity': 0.5,
                    },
                    'name': 'Contour grid',
                },
            )

        basis_labels = [
            f'σ<sub>1</sub>',
            f'σ<sub>2</sub>',
            f'σ<sub>3</sub>',
        ]
        basis_labels_unit = [f'{i}{unit}' for i in basis_labels]
        axis_labels = {
            'x': utils.format_axis_label(basis_unit[:, 0], basis_labels_unit),
            'y': utils.format_axis_label(basis_unit[:, 1], basis_labels_unit),
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
                'text': f'Equivalent stress, σ<sub>0</sub> = {fig_eq_stress_fmt}',
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
                }
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

        # TODO: sort this out:
        # Remove any NaN values:
        # valid_idx = np.logical_not(np.isnan(grid_values))
        # grid_values = grid_values[valid_idx]
        # stress_grid = stress_grid[valid_idx]

        grid_values_all = []
        for yld_func in yield_functions:
            values_i = yld_func.get_value(stress_grid * equivalent_stress)
            values_i /= equivalent_stress
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
            values_i /= equivalent_stress
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
                stress_states=None, backend='plotly'):
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
        )

    def show_2D(self, plane, normalise=True, resolution=DEF_2D_RES,
                equivalent_stress=None, min_stress=None, max_stress=None,
                stress_states=None, up=None, show_contour_grid=False):
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
        )
