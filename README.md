# MLP phonopy and phono3py workflow

This repository contains two resumable workflows that share the same fixed-cell
atomic relaxation:

1. read `POSCAR` and relax atomic positions with a fixed cell using an MLP
2. generate phonopy displacement supercells
3. evaluate displaced-supercell forces with the same MLP and write `FORCE_SETS`
4. build force constants with phonopy and write `band.yaml`

The original phonopy path is kept unchanged. The phono3py path additionally:

1. generates separate fc3 and fc2 displacement sets
2. evaluates every displaced structure with the selected MLP
3. writes `FORCES_FC3`, `FORCES_FC2`, `fc3.hdf5`, and `fc2.hdf5`
4. calculates lattice thermal conductivity using RTA, the full iterative LBTE
   solution, or both

Supported MLP calculator wrappers:

- MatterSim: `mattersim_env`
- MACE-MP: `mace_env`
- SevenNet: `sevenn_env`

## Setup

Edit `config.toml`, especially:

- `workflow.active_mlp`
- `workflow.input_poscar`
- MLP `kwargs` such as `device`, `model`, or local checkpoint paths
- `phonopy.supercell_matrix`
- `displacements.distance`
- `band.auto`, `band.nqpoints`, `band.paths`, and `band.labels`
- `phono3py.supercell_matrix` for fc3 and
  `phono3py.phonon_supercell_matrix` for fc2
- `thermal_conductivity.mesh`, `temperatures`, `sigmas`, and `methods`

All phono3py stages run in the selected MLP environment. Each of
`mattersim_env`, `mace_env`, and `sevenn_env` therefore needs phono3py installed.

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

`all` intentionally means the original phonopy chain. Run the phono3py chain
with the methods selected in `thermal_conductivity.methods` using:

```bash
python -m mlp_phonon_workflow run phono3py-all -c config.toml
```

Every phono3py stage can also be run independently:

```bash
python -m mlp_phonon_workflow run ph3-displace -c config.toml
python -m mlp_phonon_workflow run ph3-forces -c config.toml
python -m mlp_phonon_workflow run ph3-fc -c config.toml #force constant 구하기
python -m mlp_phonon_workflow run kappa-rta -c config.toml
python -m mlp_phonon_workflow run kappa-iterative -c config.toml
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
  05_phono3py_displacements/INPUT/POSCAR
  05_phono3py_displacements/OUTPUT/phono3py_disp.yaml
  05_phono3py_displacements/OUTPUT/FC3_POSCAR-00001 ...
  05_phono3py_displacements/OUTPUT/FC2_POSCAR-00001 ...
  06_phono3py_forces/INPUT/phono3py_disp.yaml
  06_phono3py_forces/INPUT/FC3_POSCAR-00001 ...
  06_phono3py_forces/OUTPUT/checkpoints/fc3/force-00001.npz ...
  06_phono3py_forces/OUTPUT/forces_fc3.npy
  06_phono3py_forces/OUTPUT/forces_fc2.npy
  07_phono3py_force_constants/INPUT/forces_fc3.npy
  07_phono3py_force_constants/INPUT/forces_fc2.npy
  07_phono3py_force_constants/OUTPUT/fc3.hdf5
  07_phono3py_force_constants/OUTPUT/fc2.hdf5
  08_kappa_rta/INPUT/{phono3py_disp.yaml,fc2.hdf5,fc3.hdf5}
  08_kappa_rta/OUTPUT/kappa-*.hdf5
  09_kappa_iterative/INPUT/{phono3py_disp.yaml,fc2.hdf5,fc3.hdf5}
  09_kappa_iterative/OUTPUT/kappa-*.hdf5
```

If a stage output already exists, that stage is skipped. Use `--force` to recompute it.
If you already have the files for a later stage, put them directly in that stage's
`INPUT/` directory and run only that stage.

The phono3py force stage writes one atomic checkpoint per displacement. If it is
interrupted, rerun the same command without `--force`; completed displacement
forces are loaded from `OUTPUT/checkpoints/`. `--force` clears those checkpoints
and recomputes all displaced structures.

For externally generated data, `ph3-fc` accepts `phono3py_disp.yaml` plus either
the workflow's `forces_fc3.npy`/`forces_fc2.npy` files or standard phono3py
`FORCES_FC3`/`FORCES_FC2` files in its `INPUT/`. The conductivity stages accept
`phono3py_disp.yaml`, `fc2.hdf5`, and `fc3.hdf5`; an optional `BORN` file in the
same directory enables NAC through phono3py.

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
