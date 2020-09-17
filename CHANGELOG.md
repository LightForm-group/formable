# Change Log

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
