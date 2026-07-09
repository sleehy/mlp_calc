# MLP phonopy workflow

This repository contains a resumable workflow:

1. read `POSCAR` and relax atomic positions with a fixed cell using an MLP
2. generate phonopy displacement supercells
3. evaluate displaced-supercell forces with the same MLP and write `FORCE_SETS`
4. build force constants with phonopy and write `band.yaml`

Supported MLP calculator wrappers:

- MatterSim: `mattersim_env`
- MACE-MP: `mace_env`
- SevenNet: `sevenn_env`

## Setup

```bash
cp config.example.toml config.toml
```

Edit `config.toml`, especially:

- `workflow.active_mlp`
- `workflow.input_poscar`
- MLP `kwargs` such as `device`, `model`, or local checkpoint paths
- `phonopy.supercell_matrix`
- `displacements.distance`
- `band.auto`, `band.nqpoints`, `band.paths`, and `band.labels`

The CLI can relaunch stages into the configured conda env automatically. You can run it
from the project root:

```bash
python -m mlp_phonon_workflow run relax -c config.toml
python -m mlp_phonon_workflow run displace -c config.toml
python -m mlp_phonon_workflow run forces -c config.toml
python -m mlp_phonon_workflow run band -c config.toml
```

Or run the whole chain:

```bash
python -m mlp_phonon_workflow run all -c config.toml
```

To switch potential without editing the config:

```bash
python -m mlp_phonon_workflow run all -c config.toml --mlp mace_mp
python -m mlp_phonon_workflow run all -c config.toml --mlp sevennet
```

## Resumable file contract

Each stage has an `INPUT/` and `OUTPUT/` directory under `runs/{mlp}`:

```text
runs/{mlp}/
  01_relax/INPUT/POSCAR
  01_relax/OUTPUT/POSCAR
  02_displacements/INPUT/POSCAR
  02_displacements/OUTPUT/phonopy_disp.yaml
  02_displacements/OUTPUT/POSCAR-001 ...
  03_forces/INPUT/phonopy_disp.yaml
  03_forces/INPUT/POSCAR-001 ...
  03_forces/OUTPUT/FORCE_SETS
  04_band/INPUT/phonopy_disp.yaml
  04_band/INPUT/FORCE_SETS
  04_band/OUTPUT/band.yaml
```

If a stage output already exists, that stage is skipped. Use `--force` to recompute it.
If you already have the files for a later stage, put them directly in that stage's
`INPUT/` directory and run only that stage.

The relax stage checks force convergence only. If it does not converge within
`relax.max_steps`, `01_relax/OUTPUT/POSCAR` is not written, so the displacement
stage cannot be populated from a failed relaxation.

By default, generated files are copied into the next stage's `INPUT/` only when that
file is missing. Set `workflow.overwrite_next_inputs = true` if you want prior stages
to refresh downstream inputs automatically.

## Status

```bash
python -m mlp_phonon_workflow status -c config.toml
```
