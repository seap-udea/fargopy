# FARGOpy
## Wrapping FRAGO3D
### Developer notes

## Commit conventions

These are the conventions for the commits:

- FEAT: Una nueva característica o funcionalidad.
- FIX: Corrección de errores o fallos.
- DOCS: Cambios relacionados con la documentación.
- STYLE: Cambios que no afectan el significado del código (espacios, formateo, etc.).
- REFACTOR: Cambios en el código que no agregan funcionalidades ni corrigen errores.
- TEST: Todo lo relacionado con pruebas y tests.
- CHORE: Tareas de mantenimiento o preparativas que no modifican ni src ni test-files.
- REL: Release of a new version (normally associated to commiting setup.py, version.py and other files changing.)

## Structure of the package

## Simulation cycle

1. Create a Simulation:
    1. If no options:
        1. Connect to the by-default FARGO3D directory ($HOME/fargo3d).
            1. Directory does not exist: raise error.
            2. Directory does not content header file: raise error.
        2. Set variables: setup, output_dir to None.
        3. Set variables: fargo3d_compilation_options
    2. If `fargo3d_dir` provided:
        1. Repeat 1.1.1
    3. If `setup` provided:
      1. Set   

1. Connect to a setup.
   - List setups
   -> Check if the setup exist.
      - No
   -> Check if the setup is locked.

---

## New Features

### Field Interpolation

- **FieldInterpolator**: Class for robust interpolation of simulation fields (scalar or vector) in 1D, 2D, and 3D.
    - Supports multiple interpolation methods: `griddata`, `RBFInterpolator`, `LinearNDInterpolator`, and Inverse Distance Weighting (IDW).
    - Allows interpolation at arbitrary coordinates and supports time interpolation between snapshots.
    - Includes domain masking to restrict interpolation to valid simulation regions.
    - Main methods:
        - `load_data`: Loads field data for specified snapshots and slices.
        - `evaluate`: Interpolates field values at given coordinates and time.
        - `create_mesh`: Generates mesh grids for arbitrary slices or the full domain.
        - `domain_mask`: Returns a boolean mask for points inside the simulation domain.

### Surface Definition and Tessellation

- **Surface**: Class to define analytical surfaces (currently spheres and cylinders) for integration and analysis.
    - Supports tessellation of surfaces into triangles (for spheres) or panels (for cylinders).
    - Attributes include centers, normals, areas, and triangles for each tessellation element.
    - Allows for hemispherical cuts via `z_cut`.
    - Main methods:
        - `tessellate_sphere`, `tessellate_cylinder`: Generate tessellated geometry.
        - `generate_dataframe`: Export tessellation data as a pandas DataFrame.
        - `calculate_all_triangle_areas`, `calculate_normals`: Compute geometric properties.

### Physical Quantities and Integrals

- **Surface methods for physical calculations**:
    - `mass_flux`: Computes the total mass flux through a surface for a range of snapshots, using interpolated density and velocity fields.
    - `total_mass`: Estimates the total mass inside a surface using Monte Carlo sampling and field interpolation.
    - Both methods support following a moving planet and updating the surface accordingly.

### Field Slicing and Visualization

- **Field**:
    - Provides slicing of fields along arbitrary directions and coordinates.
    - Supports conversion of vector fields to cartesian components.
    - `meshslice`: Returns both the sliced field and the associated coordinate mesh for plotting.
    - `plot`: Automatically detects the appropriate plane (XY, XZ, or 3D) and visualizes the field using matplotlib.

### Interactive Visualization

- **Simulation.plot_interactive**:
    - Interactive widget-based visualization for simulation outputs.
    - Allows selection of field type (density, energy, velocity), slice, resolution, and colormap.
    - Supports streamlines and overlays (e.g., Hill radius).
    - Integrates with Jupyter/IPython environments.

### Data Handling and Utilities

- **Support for precomputed simulations**:
    - Methods to list and download precomputed simulation outputs from a public repository.
- **Automatic unit handling**:
    - Simulation units can be set to CGS or MKS, with derived units for density, velocity, etc.
- **Planet class**:
    - Encapsulates planet properties and computes the Hill radius.

---

## Summary of Key Additions

- Generalized field interpolation and evaluation at arbitrary points and times.
- Analytical surface definition and tessellation for integration.
- Calculation of mass flux and total mass through/inside surfaces.
- Flexible field slicing and mesh generation for visualization and analysis.
- Interactive plotting tools for simulation data.
- Improved documentation and docstring conventions for all new classes and methods.

---



