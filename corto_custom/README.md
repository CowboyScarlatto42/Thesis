# Custom CORTO scripts

This folder contains the custom scripts developed for the thesis work on top of
the original CORTO repository.

The files are organized with the same relative layout expected inside CORTO, so
they can be copied into the root of a CORTO checkout:

```text
CORTO/
  custom_codes/
  tutorials/
```

## Contents

### `custom_codes/`

- `filter_by_phase_angle.py`: filters CORTO frames according to the Sun phase
  angle and exports the accepted frame indices.
- `poses_roe.py`: generates relative spacecraft trajectories from relative
  orbital elements.
- `sun_graph.py`: visualizes camera, body, and Sun geometry from a CORTO
  `geometry.json` file.
- `import_blender`: Blender Python helper to import CORTO camera, body, and Sun
  trajectories into an existing `.blend` scene.
- `masks_blender`: Blender Python helper to render object-index masks from an
  existing animated scene.

### `tutorials/`

- `S10_Spacecraft.py`: custom CORTO tutorial script for rendering the spacecraft
  scenario used in this work.

## Usage

Copy this folder content into the root of the original CORTO repository:

```bash
cp -R custom_codes /path/to/CORTO/
cp -R tutorials /path/to/CORTO/
```

Then run the scripts from the CORTO root, using the same Python environment and
input/output folder conventions required by CORTO.

The two Blender helpers, `import_blender` and `masks_blender`, are Python scripts
intended to be executed inside Blender or with Blender's Python interpreter.
Their hard-coded paths and object names should be adapted to the local CORTO
scenario before running them.

## Notes

This folder intentionally does not include the full CORTO repository, virtual
environments, cache files, or generated outputs. It only contains custom code
that is meant to be added to an existing CORTO checkout.

