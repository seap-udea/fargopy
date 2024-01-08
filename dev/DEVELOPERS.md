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



