# FARGOpy

## What's new

- **Version 1.2.***:
  - Added a new `coords` module (`src/fargopy/coords.py`) to improve coordinate handling workflows.
  - Improved 3D interpolation and related field-processing routines.
  - Updated core modules for analysis and post-processing: `fields`, `flux`, `plot`, `simulation`, and package initialization.
  - Added dedicated tests for coordinates/flux workflows (`test_coords_flux.py`).
  - Fixed ParaView export and ParaView data loading issues.
  - Refreshed tutorial notebooks and gallery assets for interpolation/flux examples.
  - Updated project metadata for release and citation (`.zenodo.json`, `CITATION.cff`) and refreshed README citation/badges.
  - Removed obsolete gallery assets and deprecated development notebook artifacts.

- **Version 1.0.***:
  - Refactoring of the code to make it more user-friendly.
  - Graphical interface for FARGOpy (using PyQt5).
  - Flux calculation.
  - Full documentation.
  - Tests of main modules.
  - Technical report available: [FARGOpy: A Python Package for Post-processing and Analyzing FARGO3D Hydrodynamical Simulations](https://github.com/seap-udea/fargopy/blob/main/science/introducing-fargopy/MurilloZuluagaMontesinos2026-IntroducingFARGOpy.pdf)
  - Other science papers using FARGOpy are available in the [science](https://github.com/seap-udea/fargopy/tree/main/science) directory.

- **Version 0.4.***:
  - Field interpolation in 1D, 2D, and 3D with evaluation at arbitrary points and times.
  - Analytical surface definition and tessellation for integration.
  - Calculation of mass flux and total mass through/inside surfaces.
  - Flexible field slicing and mesh generation for visualization and analysis.

- **Version 0.3.***:
  - Refactoring of initializing routines.
  - Improvements in documentation of basic classes in `__init__.py`.
  - Precomputed simulations uploaded to FARGOpy Cloud Repository and available usnig `download_precomputed` static method.

- **Version 0.2.***:
  - First real applications tested with FARGOpy.
  - All basic routines for reading output created.
  - Major refactoring. 

- **Version 0.1.***:
  - Package is now provided with a script 'ifargopy' to run 'ipython' with fargopy initialized.
  - A new 'progress' mode has been added to status method.
  - All the dynamics of loading/compiling/running/stoppìng/resuming FARGO3D has been developed.

- **Version 0.0.***:
  - First classes created.
  - The project is started!
