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

Each conductivity stage automatically creates four plots when
`thermal_conductivity.plots.enabled = true`:

- phonon density of states (DOS) vs frequency (THz)
- spectral thermal conductivity vs frequency (THz)
- cumulative thermal conductivity vs frequency (THz)
- cumulative thermal conductivity vs mean free path (log scale; nm by default)

Only the isotropic average `(kappa_xx + kappa_yy + kappa_zz) / 3` is plotted.
The DOS is calculated directly from
`runs/{mlp}/08_kappa_rta/INPUT/fc2.hdf5` and its phono3py YAML; LBTE inputs are
not used for the DOS. Conductivity curves still use their corresponding RTA or
LBTE `kappa-*.hdf5` data.
The RTA stage reads `mode_kappa`; the iterative stage reconstructs full LBTE
mode contributions and mean free paths from `f_vector`. Each source file and
temperature is retained in the CSV and metadata. Plot legends use the compact
`mlp/rta-or-lbte/NxNxN` form. RTA and LBTE inputs can be overlaid in
the same figures under one `plots/` directory. Plot settings such as the number of bins,
MFP unit, DPI, and temperatures are configured in
`[thermal_conductivity.plots]`.

For automatic stage plotting, the workflow reads every `kappa-*.hdf5` in the
current stage output directory: `runs/{mlp}/08_kappa_rta/OUTPUT/` for RTA and
`runs/{mlp}/09_kappa_iterative/OUTPUT/` for iterative LBTE (with `runs/{mlp}`
replaced by `workflow.run_dir`). The standalone `plot-kappa` command instead
reads exactly the HDF5 paths supplied as positional arguments.

Existing phono3py conductivity files can be plotted without rerunning a
conductivity calculation. Activate the conda environment used for that MLP and
run. The command reads `[thermal_conductivity.plots]` from `config.toml` by
default; use `-c` for another config, or `--bins`, `--mfp-unit`, `--dpi`, and
`--temperature` to override individual settings. For a manual one-off plot,
write outside `plot_archive` so the managed archive is not duplicated:

```bash
python -m mlp_phonon_workflow plot-kappa \
  plot_archive/mattersim/thermal_conductivity/rta/inputs/kappa-m252525.hdf5 \
  plot_archive/mattersim/thermal_conductivity/lbte/inputs/kappa-m252525.hdf5 \
  plot_archive/mace_mp/thermal_conductivity/rta/inputs/kappa-m252525.hdf5 \
  plot_archive/mace_mp/thermal_conductivity/lbte/inputs/kappa-m252525.hdf5 \
  plot_archive/sevennet/thermal_conductivity/rta/inputs/kappa-m252525.hdf5 \
  plot_archive/sevennet/thermal_conductivity/lbte/inputs/kappa-m252525.hdf5 \
  --method auto --temperature 300 --mfp-unit nm \
  --output-dir 
```

The command creates phonon DOS, spectral conductivity, cumulative conductivity
vs frequency, and cumulative conductivity vs mean free path under
`/tmp/kappa_plots/plots/`. A `combined` output directory is never used; all
explicitly supplied input cases are overlaid in the same figures, without
method-specific plot directories.
LBTE plotting
requires `f_vector`,
`group_velocity`, `heat_capacity`, `weight`, `kappa_unit_conversion`, and
`kappa` datasets. With the current phono3py writer, `f_vector` has no
temperature dimension, so an LBTE HDF5 file containing multiple temperatures
must instead be generated separately for each temperature.

### DOS with an independent mesh

`plot-kappa` reports a DOS histogram on the conductivity file's existing mesh.
For a converged harmonic DOS on a separately selected reciprocal-space mesh,
use `plot-dos`. It reads `phono3py_params.yaml` and `fc2.hdf5` from the
`ph3-fc` result, so neither MLP forces nor thermal conductivity are recomputed:

```bash
python -m mlp_phonon_workflow plot-dos --mlp mattersim --mesh 40 40 40
python -m mlp_phonon_workflow plot-dos --mlp mace_mp  --mesh 40 40 40
python -m mlp_phonon_workflow plot-dos --mlp sevennet --mesh 40 40 40
```

To calculate the same independent mesh for several MLPs and overlay their DOS
curves in one figure, list all MLPs after a single `--mlp` option:

```bash
python -m mlp_phonon_workflow plot-dos \
  --mlp mattersim mace_mp sevennet \
  --mesh 40 40 40
```

Each MLP keeps its own reproducible DOS output. The comparison is written to
`plot_archive/plots/dos/mesh-40x40x40/` and does not create a `combined`
directory. Use `--output-dir` to select another comparison output root.

Each command automatically relaunches into that MLP's configured conda
environment. Tetrahedron integration is the default. Gaussian smearing is also
available, for example `--method gaussian --sigma 0.1`. The default mesh and
integration settings are configurable independently of the IFC supercell and
the conductivity mesh:

```toml
[dos]
enabled = true
mesh = [40, 40, 40]
method = "tetrahedron"
is_gamma_center = true
is_mesh_symmetry = true
dpi = 200
```

After every `ph3-fc` completion or skip, the DOS inputs are archived and the
configured DOS mesh is generated automatically under each MLP's own directory.

## Plot archive

Files needed to reproduce band, DOS, and thermal-conductivity plots are copied
automatically from each stage `OUTPUT/` into a simpler archive tree when
`plot_archive.enabled = true`:

```text
plot_archive/{mlp}/
  band/inputs/
    band.yaml
    phonopy_params.yaml
  dos/
    inputs/{phono3py_params.yaml,fc2.hdf5}
    mesh-NxNxN/
      inputs/{phono3py_params.yaml,fc2.hdf5}
      {phonon_dos.png,phonon_dos.csv,metadata.json}
  thermal_conductivity/
    rta/
      inputs/kappa-*.hdf5
    lbte/
      inputs/kappa-*.hdf5
    plots/{phonon_dos,spectral_thermal_conductivity,...}.{png,csv}

plot_archive/plots/dos/mesh-NxNxN/
  {phonon_dos.png,phonon_dos.csv,metadata.json}
```

Band files are synchronized after the `band` stage. Conductivity HDF5 files are
synchronized after `kappa-rta` or `kappa-iterative`, and the DOS and conductivity
plots are regenerated from all RTA and LBTE HDF5 files currently archived for
that MLP. No combined archive
directory is generated; every plot and metadata file remains under its source
MLP directory. Synchronization runs when an already completed stage is skipped,
so it can be applied to older run directories without recomputing the
calculation. Configure the location and update behavior with:

```toml
[plot_archive]
enabled = true
root = "plot_archive"
overwrite = true
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

## Code layout

The workflow keeps `stages.py` as a small public façade and separates each
responsibility into focused modules:

- `phonopy_stages.py`: relaxation, harmonic displacements, forces, and bands
- `phono3py_stages.py`: third-order displacements, forces, IFCs, and conductivity
- `stage_archive.py`: band, DOS, and conductivity plot archives
- `stage_common.py`: stage paths, input propagation, and shared file helpers
- `phono3py_utils.py`: displacement manifests and resumable force checkpoints
