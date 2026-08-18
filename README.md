# MLP phonopy and phono3py workflow

This repository contains two resumable workflows that share the same atomic
relaxation:

1. read `POSCAR` and relax atomic positions, optionally together with the cell,
   using an MLP
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
- `relax.cell.enabled` and `relax.cell.mask` for cell relaxation
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

Plot the archived `band.yaml` as a calculated dispersion relation without
experimental data:

```bash
python -m mlp_phonon_workflow plot-band \
  -c config.toml \
  --mode dispersion
```

To overlay a two-column experimental CSV, select `experiment` mode. When the
digitized x axis uses custom high-symmetry-point positions, give one position
for the path start and every segment end:

```bash
python -m mlp_phonon_workflow plot-band \
  -c config.toml \
  --mode experiment \
  --experiment-csv experiment_points.csv \
  --high-symmetry-positions 0 0.3048762 0.4124713 0.736312 1
```

Without `--high-symmetry-positions`, calculated cumulative band distances are
normalized to 0--1. Use `--band-yaml` for a non-archived input and `--output`
to select the PNG path. Imaginary modes remain at their negative signed
frequencies in both modes. The existing `run all` command is unchanged and
does not select a plotting mode.

Override the configured run directory for one command with `--run-dir`. The
override is retained when the command automatically relaunches into an MLP
conda environment:

```bash
python -m mlp_phonon_workflow run relax \
  -c config.toml \
  --poscar mlp_phonon_workflow/POSCAR_FA_100 \
  --run-dir runs/sevennet/cubic/FA_100
```

### Atomic and cell relaxation

The relax stage always optimizes the atomic degrees of freedom allowed by the
workflow constraints. Any `Selective dynamics` flags inherited from the source
POSCAR are removed from the copied `01_relax/INPUT/POSCAR` before the
code-defined fixed-atom and fixed-bond constraints are applied. The original
source POSCAR is not modified. Cell relaxation is optional:

```toml
[relax]
optimizer = "BFGS"
fmax = 0.001
max_steps = 500
maxstep = 0.2  # maximum optimizer displacement per iteration in angstrom
write_extxyz = true

[relax.cell]
enabled = true
filter = "frechet"  # recommended; "unit" is also available
mask = [true, true, true, true, true, true]  # xx, yy, zz, yz, xz, xy
hydrostatic_strain = false
constant_volume = false
scalar_pressure_gpa = 0.0

[relax.symmetry]
enabled = true
symprec = 1.0e-3
adjust_positions = true
adjust_cell = true
verbose = false

[relax.n_h_bond_restraint]
enabled = false
cutoff = 1.25  # angstrom
spring_constant = 50.0  # eV/angstrom^2
```

With `enabled = false`, the cell is fixed and only atomic positions are
relaxed. With `enabled = true`, the selected MLP must provide stress. The
Frechet cell filter minimizes atomic and cell degrees of freedom together;
`fmax` is applied to their combined generalized forces. Set
`hydrostatic_strain = true` to allow only isotropic volume changes, or use
`mask` to select individual strain components. A positive
`scalar_pressure_gpa` applies compressive external pressure.
The `maxstep` setting limits the optimizer displacement in one iteration and
defaults to ASE's standard value of 0.2 angstrom.

When `relax.symmetry.enabled = true`, ASE `FixSymmetry` symmetrizes atomic
forces, stress, atomic steps, and cell-deformation steps throughout the
optimization. For example, a `P4bm` tetragonal input retains `a=b` during cell
relaxation. `symprec` controls the symmetry detected from the input structure.
If the input is already detected as `P1`, the workflow prints a warning because
there is no non-trivial symmetry to preserve. Start from the symmetric
structure rather than from an already symmetry-broken relaxation when the
crystal phase must remain fixed.

Set `relax.n_h_bond_restraint.enabled = true` to apply
`0.5 * k * (r - r0)^2` to every initial N-H bond found within `cutoff`,
including bonds crossing a periodic boundary. Here `r0` is each initial bond
length and `spring_constant` is `k` in eV/angstrom^2. The restraint contributes
energy, forces, and virial stress, so it can also be used during cell
relaxation. The detected zero-based atom-index pairs and target lengths are
recorded in `metadata.json`. Enabling the option without a matching N-H pair
raises an error so that an unsuitable cutoff does not silently run
unrestrained.

The output `metadata.json` records the initial and final cells and volumes,
the constrained and unconstrained maximum atomic forces, and the final stress
and pressure when cell relaxation is enabled.

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

Set `dos.projected = true` or pass `--projected` to additionally calculate
element-projected phonon DOS. The total DOS remains in `phonon_dos.*`; the
projected result is written separately as `phonon_projected_dos.png` and
`phonon_projected_dos.csv`:

```bash
python -m mlp_phonon_workflow plot-dos \
  --mlp sevennet \
  --mesh 40 40 40 \
  --projected
```

The projected CSV contains the total DOS and one column per element in the
primitive cell. Phonopy requires eigenvectors on the full q-point mesh for
projected DOS, so mesh symmetry is automatically disabled for this calculation.
This uses more memory and time than total DOS. Use `--no-projected` to override
an enabled config for a single command.
When multiple MLPs are selected, each MLP gets its own projected-DOS files;
the shared comparison figure continues to overlay total DOS curves only.

To calculate total and projected DOS directly from the harmonic phonopy stage,
pass its `phonopy_params.yaml`. This file contains the displacement-force
dataset, so the workflow reconstructs the harmonic force constants without
using phono3py:

```bash
python -m mlp_phonon_workflow plot-dos \
  -c config.toml \
  --mlp sevennet \
  --phonopy-yaml runs/sevennet/cubic/FA_100/04_band/OUTPUT/phonopy_params.yaml \
  --mesh 40 40 40 \
  --projected
```

By default, this writes to
`<phonopy-yaml-directory>/dos/mesh-NxNxN/`. Use `--output-dir` to select an
exact output directory. If the YAML does not contain forces, provide an
existing `FORCE_CONSTANTS` or `force_constants.hdf5` with
`--force-constants`; an optional NAC file can be supplied with `--born`.
`band.yaml` alone is not a valid DOS input because it only contains the
one-dimensional band path.

To draw the harmonic band structure and element-projected DOS side by side,
use `plot-band-dos`:

```bash
python -m mlp_phonon_workflow plot-band-dos \
  -c config.toml \
  --mlp sevennet \
  --phonopy-yaml runs/sevennet/cubic/FA_100/04_band/OUTPUT/phonopy_params.yaml \
  --mesh 40 40 40
```

This command uses Phonopy's built-in
`phonon.plot_band_structure_and_dos(pdos_indices=...)`. Primitive-cell atoms
are grouped by element and passed as zero-based PDOS index groups. The band
path comes from the `[band]` config (`auto`, `nqpoints`, `paths`, and
`labels`), while the mesh and integration defaults come from `[dos]`.
The default outputs beside the input YAML are
`phonon_band_projected_dos.png` and `phonon_band_projected_dos.json`.
Use `--output` to select another PNG path; the metadata JSON is written with
the same basename.

To customize the Matplotlib Figure while keeping the CLI workflow, write a
Python file that defines `customize(fig)`:

```python
# customize_band_pdos.py
def customize(fig):
    band_axes = [
        axis for axis in fig.axes
        if (axis.get_gid() or "").startswith("band_")
    ]
    pdos_axis = next(
        axis for axis in fig.axes
        if axis.get_gid() == "projected_dos"
    )

    for axis in [*band_axes, pdos_axis]:
        axis.set_ylim(-5, 20)
    band_axes[0].set_ylabel("Phonon frequency (THz)")
    pdos_axis.set_xlabel("Element-projected DOS")
    fig.suptitle("Customized band structure and PDOS")
```

Pass that file to the workflow:

```bash
python -m mlp_phonon_workflow plot-band-dos \
  -c config.toml \
  --phonopy-yaml runs/sevennet/cubic/FA_100/04_band/OUTPUT/phonopy_params.yaml \
  --mesh 40 40 40 \
  --customizer customize_band_pdos.py
```

For full Matplotlib control from another Python file, use the Figure-returning
API. It performs the Phonopy calculations and built-in combined plotting but
does not save or close the figure:

```python
import matplotlib.pyplot as plt

from mlp_phonon_workflow.dos_plot import (
    plot_band_with_projected_dos_from_phonopy,
)

figure = plot_band_with_projected_dos_from_phonopy(
    "runs/sevennet/cubic/FA_100/04_band/OUTPUT/phonopy_params.yaml",
    mesh=[40, 40, 40],
)

band_axes = [
    axis for axis in figure.axes
    if (axis.get_gid() or "").startswith("band_")
]
pdos_axis = next(
    axis for axis in figure.axes
    if axis.get_gid() == "projected_dos"
)

for axis in [*band_axes, pdos_axis]:
    axis.set_ylim(-5, 20)
band_axes[0].set_ylabel("Phonon frequency (THz)")
pdos_axis.set_xlabel("Element-projected DOS")
figure.suptitle("Customized band structure and PDOS")

figure.savefig("custom_band_pdos.png", dpi=300, bbox_inches="tight")
plt.close(figure)
```

The returned object is a `matplotlib.figure.Figure`. Populated band axes have
IDs `band_0`, `band_1`, and so on, while the PDOS axis has ID
`projected_dos`, making them easy to select without relying on axes order.

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
projected = true
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
      {phonon_dos.png,phonon_dos.csv}
      {phonon_projected_dos.png,phonon_projected_dos.csv,metadata.json}
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

The fixed-cell relax stage checks atomic force convergence. With cell
relaxation enabled, convergence includes the generalized cell forces from the
selected cell filter. If it does not converge within `relax.max_steps`,
`01_relax/OUTPUT/POSCAR` is not written, so the displacement stage cannot be
populated from a failed relaxation.

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
