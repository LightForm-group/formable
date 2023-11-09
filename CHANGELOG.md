# Change Log

## [0.1.21] - 2023.11.09

### Fixed

- Resolve numpy deprecation.

## [0.1.20] - 2022.08.08

### Changed

- Constrain LM fitting process
- Relabel attribute `true_stress` to `stress` in `LoadResponse` class

## [0.1.19] - 2021.11.09

### Added

- Add `cyclic_uniaxial` load case method to task `generate_load_case`
- Add `mixed` load case method to task `generate_load_case`

## [0.1.18] - 2021.08.06

### Added

- Add option `include_yield_functions` to `LoadResponseSet.show_yield_functions_2D` and `LoadResponseSet.show_yield_functions_3D`, which is a list of fitted yield function indices to include in the visualisation.
- Add `get_load_case_planar_2D` load case function.
- Add option `strain_rate_mode` to `get_load_case_plane_strain`, which determines if the load case is defined by deformation gradient (`F_rate`), velocity gradient (`L`) or an approximation excluding the stress condition (`L_approx`), which is useful when we want to avoid using mixed boundary conditions.

### Changed

- Functions `get_load_case_uniaxial`, `get_load_case_biaxial` and `get_load_case_plane_strain` have been refactored, documented and generalised where applicable. The returned `dict` from these functions now includes passing through `direction` and `rotation`. A new key `rotation_matrix` is the matrix representation of the rotation specified, if specified.

## [0.1.17] - 2021.05.11

### Added

- Add animation widget for yield func evolution: `animate_yield_function_evolution(load_response_sets, ...)`.
- Ability to add sheet direction labels in yield function plots.

## [0.1.16] - 2021.04.23

### Changed

- 2D yield function plotting now use scikit-image to compute the zero-contour, which can then be plotted as scatter-plot data, instead of a Plotly contour plot. Partial fix (2D only) for [#9](https://github.com/LightForm-group/formable/issues/9). The old behaviour can be switched on with `use_plotly_contour=True` in methods: `YieldFunction.compare_2D`, `YieldFunction.show_2D` and `LoadResponseSet.show_yield_functions_2D`.
- The `YieldFunction.yield_point` attribute is saved in the dict representation of a each `LoadResponseSet.fitted_yield_function`, and loaded correctly when loading from a dict, via a change to `YieldFunction.from_name`.
- Parameter fitting using the `LMFitter` class now scales the fitting parameters to one.

## [0.1.15] - 2021.04.10

### Fixed

- Bug fix in `LoadResponseSet.to_dict` if an associated yield function was not fitted.

## [0.1.14] - 2021.04.10

### Added

- Add ability to specify fitting bounds and other optimisation parameters in `YieldFunction.from_fit` and `LoadResponseSet.fit_yield_function`.

### Changed

- `LoadResponseSet.yield_functions` attribute renamed `LoadResponseSet.fitted_yield_functions`.

## [0.1.13] - 2021.03.28

### Changed

- Do not modify input dict to `levenberg_marquardt.LMFitter.from_dict`.
- Fix bug in `TensileTest.show()` stress scale.

### Added

- Add `to_dict` and `from_dict` methods to `LoadResponseSet`.

## [0.1.12] - 2020.12.16

### Added

- Add `LMFitter.from_dict`

### Fixed

- Add `single_crystal_parameters` to returned dict of `LMFitter.to_dict`.

## [0.1.11] - 2020.12.16

### Fixed

- Set float values in `get_new_single_crystal_params`.

## [0.1.10] - 2020.12.15

### Added

- Add new module, `levenberg_marquardt` for fitting single crystal parameters.

## [0.1.9] - 2020.11.18

### Fixed

- Add missing import to `formable.utils`.

## [0.1.8] - 2020.11.18

### Added

- Include `tensile_test` module from `tensile_test` package.

## [0.1.7] - 2020.09.17

### Fixed

- Fix plot line colouring for many traces (more than Plotly default colour list)

## [0.1.6] - 2020.08.22

### Changed

- Add `dump_frequency` to load case generators.

## [0.1.5] - 2020.08.18

### Changed

- Default tolerance for `LoadResponse.is_uniaxial` check loosened to 0.3.

## [0.1.4] - 2020.07.01

### Changed

- Print out the degree to which the stress state is uniaxial in `LoadResponse.is_uniaxial`.

## [0.1.3] - 2020.06.09

### Added

- Add a method to estimate the Lankford coefficient via the tangent of the yield surface at a uniaxial stress state: `YieldFunction.get_numerical_lankford`
- Add options to `YieldFunction.show_2D`, `YieldFunction.compare_2D` and `LoadResponseSet.show_yield_functions_2D` to visualise the tangent and normal to the yield function at a uniaxial stress state.
- Add incremental data: `equivalent_plastic_strain` and `accumulated_shear_strain`, and associated `YieldPointCriteria` mappings for getting the yield stress (using the same method as that used for `equivalent_stress` [total]).
- Add `show_stress_states` to `LoadResponseSet.show_yield_functions_3D` and `LoadResponseset.show_yield_functions_2D` to optionally hide stress states.
- Add option to pass Plotly `layout` parameters to yield function visualisation methods.
- Add property `num_increments` to `LoadResponse`.
- Add `repr` to `LoadResponse` and `LoadResponseSet`.
- Add `YieldFunction.from_name()` class method for generating a yield function from a string name and parameters.
- Add `LoadResponse.incremental_data` property to return all incremental data as a `dict`.

### Changed

- Check each `incremental_data` array passed to `LoadResponse` has the same outer shape (i.e. same number of increments).
- `AVAILABLE_YIELD_FUNCTIONS` and `YIELD_FUNCTION_MAP` have been replaced with functions `get_available_yield_functions` and `get_yield_function_map`, respectively.
- Number of excluded load responses is printed when performing yield function fitting.

## [0.1.2] - 2020.05.09

### Fixed

- Fixed an issue when visualising yield surfaces in 3D (via `YieldSurface.compare_3D()`) (and also 2D) where, if the value of the yield function residual was already normalised (e.g. by the equivalent stress), then the isosurface drawn by Plotly was defective (showing spikes beyond the bounds of the contour grid), since the values that were being contoured were of the order 10^-8. This was because we normalised by the equivalent stress again when calculating the contour values. This was fixed by normalising by the absolute maximum value in the values that are returned by the residual function, rather than always normalising by the equivalent stress, so the contour values should be of the order 1 now, regardless of whether a given yield function residual value is normalised or not.
- Fixed yield function residual for `Barlat_Yld91`, where hydrostatic stresses would return `np.nan`.
- Check for bad `kwargs` in `LoadResponseSet.fit_yield_function`.
- Added an `equivalent_stress` parameter to `Hill1948` to make it fit and visualise like the others. Not sure if this is the correct approach.

### Added

- Added an option to show the bounds of the 3D contour grid when visualising yield functions in 3D.
- Added an option to associate additional text in visualising yield functions (for the legend): `legend_text`.
- Added module `load_cases` for generating load cases for simulations.
- Added hover text in `YieldFunction.compare_2D` that shows the value(s) of the yield function at each grid point.
- Added `lankford` property to `Hill1948` that returns the Lankford coefficient, as determined by the values of the anisotropic parameters.

### Changed

- The tolerance for checking if a `uniaxial_response` passed to `LoadResponseSet.fit_yield_function` is in fact uniaxial has been loosened, since this way failing when it shouldn't have.
- Normalise all yield function residuals by their equivalent stress parameter.

## [0.1.1] - 2020.04.12

### Changed

Image URLs in README

## [0.1.0] - 2020.04.12

Initial release.
