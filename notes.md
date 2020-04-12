
### Notes
  - TODO: add `to/from_json`
  - `matflow` integration:
      - `matflow-formable` will provide interface
  - `TensileTest` could subclass `LoadResponse`? Could have `dimension` attribute, which
    is the max dimension considered in any analysis (1, 2 or 3). Then the
    `LoadResponse.incremental_data` outer shape would be of that size.
