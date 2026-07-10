from __future__ import annotations

import filecmp
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .calculators import build_calculator
from .config import input_poscar, run_dir


PHONOPY_STAGES = ("relax", "displace", "forces", "band")
PHONO3PY_ONLY_STAGES = (
    "ph3-displace",
    "ph3-forces",
    "ph3-fc",
    "kappa-rta",
    "kappa-iterative",
)
STAGES = PHONOPY_STAGES + PHONO3PY_ONLY_STAGES

STAGE_DIR_NAMES = {
    "relax": "01_relax",
    "displace": "02_displacements",
    "forces": "03_forces",
    "band": "04_band",
    "ph3-displace": "05_phono3py_displacements",
    "ph3-forces": "06_phono3py_forces",
    "ph3-fc": "07_phono3py_force_constants",
    "kappa-rta": "08_kappa_rta",
    "kappa-iterative": "09_kappa_iterative",
}

EXPECTED_OUTPUTS = {
    "relax": ("POSCAR",),
    "displace": ("phonopy_disp.yaml",),
    "forces": ("FORCE_SETS",),
    "band": ("band.yaml",),
    "ph3-displace": ("phono3py_disp.yaml", "displacement_manifest.json"),
    "ph3-forces": ("forces_fc3.npy", "forces_fc2.npy", "completed.json"),
    "ph3-fc": ("fc3.hdf5", "fc2.hdf5", "completed.json"),
    "kappa-rta": ("completed.json",),
    "kappa-iterative": ("completed.json",),
}


@dataclass(frozen=True)
class StagePaths:
    base: Path
    input: Path
    output: Path


def stage_paths(config: dict[str, Any], stage: str) -> StagePaths:
    base = run_dir(config) / STAGE_DIR_NAMES[stage]
    return StagePaths(base=base, input=base / "INPUT", output=base / "OUTPUT")


def run_stage(config: dict[str, Any], stage: str, *, force: bool = False) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    for name in STAGES:
        paths = stage_paths(config, name)
        paths.input.mkdir(parents=True, exist_ok=True)
        paths.output.mkdir(parents=True, exist_ok=True)

    if stage == "relax":
        _run_relax(config, force=force)
    elif stage == "displace":
        _run_displace(config, force=force)
    elif stage == "forces":
        _run_forces(config, force=force)
    elif stage == "band":
        _run_band(config, force=force)
    elif stage == "ph3-displace":
        _run_ph3_displace(config, force=force)
    elif stage == "ph3-forces":
        _run_ph3_forces(config, force=force)
    elif stage == "ph3-fc":
        _run_ph3_fc(config, force=force)
    elif stage == "kappa-rta":
        _run_thermal_conductivity(config, method="rta", force=force)
    elif stage == "kappa-iterative":
        _run_thermal_conductivity(config, method="iterative", force=force)


def stage_status(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for stage in STAGES:
        paths = stage_paths(config, stage)
        expected = [paths.output / name for name in EXPECTED_OUTPUTS[stage]]
        rows.append(
            {
                "stage": stage,
                "input": str(paths.input),
                "output": ", ".join(str(path) for path in expected),
                "status": "done" if all(path.exists() for path in expected) else "pending",
            }
        )
    return rows


def _run_relax(config: dict[str, Any], *, force: bool) -> None:
    paths = stage_paths(config, "relax")
    in_poscar = paths.input / "POSCAR"
    if not in_poscar.exists():
        _copy_file(input_poscar(config), in_poscar, overwrite=False)

    out_poscar = paths.output / "POSCAR"
    if out_poscar.exists() and not force:
        print(f"[relax] skip: {out_poscar} exists")
        _populate_displace_input(config)
        _populate_ph3_displace_input(config)
        return

    if force:
        _clear_matching_displace_input(config, out_poscar)
        _clear_matching_ph3_displace_input(config, out_poscar)
        _clear_relax_success_outputs(paths)

    from ase.io import read, write
    from ase.optimize import BFGS, FIRE, LBFGS

    atoms = read(in_poscar, format="vasp")
    atoms.calc = build_calculator(config)

    relax_config = config["relax"]
    optimizer_name = str(relax_config.get("optimizer", "LBFGS")).upper()
    optimizers = {"LBFGS": LBFGS, "BFGS": BFGS, "FIRE": FIRE}
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown relax.optimizer: {optimizer_name}")

    trajectory = str(paths.output / "relax.traj")
    logfile = str(paths.output / "relax.log")
    optimizer = optimizers[optimizer_name](atoms, trajectory=trajectory, logfile=logfile)
    converged = bool(
        optimizer.run(
            fmax=float(relax_config.get("fmax", 0.01)),
            steps=int(relax_config.get("max_steps", 500)),
        )
    )

    metadata = {
        "stage": "relax",
        "input": str(in_poscar),
        "output": str(out_poscar),
        "converged": converged,
        "cell_fixed": True,
        "max_force_ev_per_a": _safe_float(lambda: _max_force(atoms)),
        "potential_energy_ev": _safe_float(lambda: atoms.get_potential_energy()),
    }

    if not converged:
        _clear_relax_success_outputs(paths)
        metadata["output"] = None
        _write_json(paths.output / "metadata.json", metadata)
        raise RuntimeError(
            "Atomic-position relaxation did not reach force convergence within "
            f"{relax_config.get('max_steps')} steps. No relaxed POSCAR was written."
        )

    write(out_poscar, atoms, format="vasp", direct=True, vasp5=True)
    write(paths.output / "CONTCAR", atoms, format="vasp", direct=True, vasp5=True)
    if bool(relax_config.get("write_extxyz", True)):
        write(paths.output / "relaxed.extxyz", atoms, format="extxyz")

    _write_json(paths.output / "metadata.json", metadata)
    _populate_displace_input(config)
    _populate_ph3_displace_input(config)


def _run_displace(config: dict[str, Any], *, force: bool) -> None:
    _populate_displace_input(config)
    paths = stage_paths(config, "displace")
    in_poscar = paths.input / "POSCAR"
    if not in_poscar.exists():
        raise FileNotFoundError(
            f"Missing {in_poscar}. Put a POSCAR there or run the relax stage first."
        )

    out_yaml = paths.output / "phonopy_disp.yaml"
    if out_yaml.exists() and not force:
        print(f"[displace] skip: {out_yaml} exists")
        _populate_forces_input(config)
        return

    from phonopy import Phonopy
    from phonopy.interface.calculator import (
        read_crystal_structure,
        write_crystal_structure,
        write_supercells_with_displacements,
    )

    phonopy_config = config["phonopy"]
    interface_mode = phonopy_config.get("calculator", "vasp")
    unitcell, structure_info = read_crystal_structure(
        filename=in_poscar,
        interface_mode=interface_mode,
    )
    if unitcell is None:
        raise RuntimeError(f"phonopy could not read structure: {in_poscar}")

    phonon = Phonopy(
        unitcell,
        supercell_matrix=phonopy_config.get("supercell_matrix", [2, 2, 2]),
        primitive_matrix=phonopy_config.get("primitive_matrix", "auto"),
        symprec=float(phonopy_config.get("symprec", 1.0e-5)),
        is_symmetry=bool(phonopy_config.get("is_symmetry", True)),
        use_SNF_supercell=bool(phonopy_config.get("use_SNF_supercell", False)),
        calculator=interface_mode,
    )

    displacement_kwargs = _filtered_kwargs(
        config["displacements"],
        {
            "distance",
            "is_plusminus",
            "is_diagonal",
            "is_trigonal",
            "number_of_snapshots",
            "random_seed",
            "temperature",
            "cutoff_frequency",
            "max_distance",
            "number_estimation_factor",
        },
    )
    phonon.generate_displacements(**displacement_kwargs)
    phonon.save(filename=out_yaml)

    write_crystal_structure(paths.output / "POSCAR", unitcell, interface_mode=interface_mode)
    write_supercells_with_displacements(
        interface_mode,
        phonon.supercell,
        phonon.supercells_with_displacements,
        optional_structure_info=structure_info,
        zfill_width=int(config["displacements"].get("zfill_width", 3)),
        additional_info={
            "pre_filename": str(
                paths.output / config["displacements"].get("supercell_filename_prefix", "POSCAR")
            )
        },
    )

    _write_json(
        paths.output / "metadata.json",
        {
            "stage": "displace",
            "input": str(in_poscar),
            "phonopy_disp_yaml": str(out_yaml),
            "n_displacements": len(phonon.supercells_with_displacements),
        },
    )
    _populate_forces_input(config)


def _run_forces(config: dict[str, Any], *, force: bool) -> None:
    _populate_forces_input(config)
    paths = stage_paths(config, "forces")
    in_yaml = paths.input / "phonopy_disp.yaml"
    if not in_yaml.exists():
        raise FileNotFoundError(
            f"Missing {in_yaml}. Put phonopy_disp.yaml there or run displace first."
        )

    out_force_sets = paths.output / "FORCE_SETS"
    if out_force_sets.exists() and not force:
        print(f"[forces] skip: {out_force_sets} exists")
        _populate_band_input(config)
        return

    import phonopy
    import numpy as np
    from ase.io import read
    from phonopy.file_IO import write_FORCE_SETS

    phonon = phonopy.load(
        phonopy_yaml=in_yaml,
        calculator=config["phonopy"].get("calculator", "vasp"),
        produce_fc=False,
        symprec=float(config["phonopy"].get("symprec", 1.0e-5)),
    )
    supercells = list(phonon.supercells_with_displacements)
    if not supercells:
        raise RuntimeError(f"No displaced supercells found in {in_yaml}")

    poscar_files = _displacement_poscars(paths.input)
    use_poscar_files = len(poscar_files) == len(supercells)
    if poscar_files and not use_poscar_files:
        raise RuntimeError(
            f"Found {len(poscar_files)} POSCAR-* files but phonopy expects "
            f"{len(supercells)} displacements."
        )

    calculator = build_calculator(config)
    force_sets = []
    energies = []
    for index, supercell in enumerate(supercells, start=1):
        if use_poscar_files:
            atoms = read(poscar_files[index - 1], format="vasp")
            source = poscar_files[index - 1].name
        else:
            atoms = _phonopy_atoms_to_ase(supercell)
            source = "phonopy_disp.yaml"
        atoms.calc = calculator
        forces = np.asarray(atoms.get_forces(), dtype=float)
        drift = forces.mean(axis=0)
        print(
            f"[forces] drift before correction = "
            f"[{drift[0]: .3e}, {drift[1]: .3e}, {drift[2]: .3e}] eV/A, "
            f"norm = {np.linalg.norm(drift):.3e} eV/A"
        )

        if bool(config["forces"].get("subtract_drift", True)):
            forces = forces - drift[None, :]
        force_sets.append(forces)
        energies.append(_safe_float(lambda atoms=atoms: atoms.get_potential_energy()))
        print(f"[forces] {index:04d}/{len(supercells):04d} from {source}")

    forces_array = np.asarray(force_sets, dtype=float)
    phonon.forces = forces_array
    write_FORCE_SETS(phonon.dataset, filename=out_force_sets)
    phonon.save(filename=paths.output / "phonopy_params.yaml")

    if bool(config["forces"].get("write_npz", True)):
        energy_array = np.asarray(
            [np.nan if energy is None else energy for energy in energies],
            dtype=float,
        )
        np.savez_compressed(
            paths.output / "forces.npz",
            forces=forces_array,
            energies=energy_array,
        )

    _write_json(
        paths.output / "metadata.json",
        {
            "stage": "forces",
            "input": str(in_yaml),
            "output": str(out_force_sets),
            "n_displacements": len(force_sets),
        },
    )
    _populate_band_input(config)


def _run_band(config: dict[str, Any], *, force: bool) -> None:
    _populate_band_input(config)
    paths = stage_paths(config, "band")
    in_yaml = paths.input / "phonopy_disp.yaml"
    in_force_sets = paths.input / "FORCE_SETS"
    if not in_yaml.exists():
        raise FileNotFoundError(f"Missing {in_yaml}.")
    if not in_force_sets.exists():
        raise FileNotFoundError(f"Missing {in_force_sets}.")

    out_band = paths.output / "band.yaml"
    if out_band.exists() and not force:
        print(f"[band] skip: {out_band} exists")
        return

    import phonopy
    from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5
    from phonopy.phonon.band_structure import get_band_qpoints

    band_config = config["band"]
    fc_options = str(band_config.get("fc_calculator_options", "")).strip() or None
    fc_calculator = str(band_config.get("fc_calculator", "traditional")).strip() or None
    phonon = phonopy.load(
        phonopy_yaml=in_yaml,
        force_sets_filename=in_force_sets,
        calculator=config["phonopy"].get("calculator", "vasp"),
        produce_fc=True,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_options,
        symprec=float(config["phonopy"].get("symprec", 1.0e-5)),
    )

    nqpoints = _band_nqpoints(band_config)
    if bool(band_config.get("auto", True)):
        phonon.auto_band_structure(
            npoints=nqpoints,
            with_eigenvectors=bool(band_config.get("with_eigenvectors", False)),
            with_group_velocities=bool(band_config.get("with_group_velocities", False)),
            write_yaml=True,
            filename=out_band,
        )
    else:
        paths_config = band_config.get("paths", [])
        if not paths_config:
            raise ValueError("band.auto=false requires band.paths.")
        band_paths = get_band_qpoints(paths_config, nqpoints)
        phonon.run_band_structure(
            band_paths,
            with_eigenvectors=bool(band_config.get("with_eigenvectors", False)),
            with_group_velocities=bool(band_config.get("with_group_velocities", False)),
            is_band_connection=bool(band_config.get("is_band_connection", False)),
            labels=band_config.get("labels") or None,
        )
        phonon.write_yaml_band_structure(filename=out_band)

    if bool(band_config.get("write_force_constants", False)):
        fc_format = str(band_config.get("force_constants_format", "hdf5")).lower()
        if fc_format == "hdf5":
            write_force_constants_to_hdf5(
                phonon.force_constants,
                filename=str(paths.output / "force_constants.hdf5"),
            )
        elif fc_format == "text":
            write_FORCE_CONSTANTS(
                phonon.force_constants,
                filename=paths.output / "FORCE_CONSTANTS",
            )
        else:
            raise ValueError("band.force_constants_format must be 'hdf5' or 'text'.")

    phonon.save(filename=paths.output / "phonopy_params.yaml")
    _write_json(
        paths.output / "metadata.json",
        {
            "stage": "band",
            "input_force_sets": str(in_force_sets),
            "input_phonopy_yaml": str(in_yaml),
            "output": str(out_band),
        },
    )


def _run_ph3_displace(config: dict[str, Any], *, force: bool) -> None:
    _populate_ph3_displace_input(config)
    paths = stage_paths(config, "ph3-displace")
    in_poscar = paths.input / "POSCAR"
    if not in_poscar.exists():
        raise FileNotFoundError(
            f"Missing {in_poscar}. Put a POSCAR there or run the relax stage first."
        )

    if _stage_complete(config, "ph3-displace") and not force:
        print(f"[ph3-displace] skip: outputs already exist in {paths.output}")
        _populate_ph3_forces_input(config)
        return

    _clear_ph3_displacement_outputs(paths.output)

    from phono3py import Phono3py
    from phonopy.interface.calculator import (
        read_crystal_structure,
        write_crystal_structure,
    )

    ph3_config = config["phono3py"]
    interface_mode = ph3_config.get("calculator", "vasp")
    unitcell, structure_info = read_crystal_structure(
        filename=in_poscar,
        interface_mode=interface_mode,
    )
    if unitcell is None:
        raise RuntimeError(f"phono3py could not read structure: {in_poscar}")

    phonon_supercell_matrix = ph3_config.get("phonon_supercell_matrix")
    if phonon_supercell_matrix == []:
        phonon_supercell_matrix = None
    ph3 = Phono3py(
        unitcell,
        supercell_matrix=ph3_config.get("supercell_matrix", [2, 2, 2]),
        primitive_matrix=ph3_config.get("primitive_matrix", "auto"),
        phonon_supercell_matrix=phonon_supercell_matrix,
        cutoff_frequency=float(ph3_config.get("cutoff_frequency", 1.0e-4)),
        is_symmetry=bool(ph3_config.get("is_symmetry", True)),
        is_mesh_symmetry=bool(ph3_config.get("is_mesh_symmetry", True)),
        use_grg=bool(ph3_config.get("use_grg", False)),
        make_r0_average=bool(ph3_config.get("make_r0_average", True)),
        symprec=float(ph3_config.get("symprec", 1.0e-5)),
        calculator=interface_mode,
        log_level=int(ph3_config.get("log_level", 1)),
        lang=str(ph3_config.get("lang", "Rust")),
    )

    displacement_config = ph3_config["displacements"]
    fc3_config = displacement_config.get("fc3", {})
    fc2_config = displacement_config.get("fc2", {})
    ph3.generate_displacements(
        **_filtered_kwargs(
            fc3_config,
            {
                "distance",
                "cutoff_pair_distance",
                "is_plusminus",
                "is_diagonal",
                "number_of_snapshots",
                "random_seed",
                "max_distance",
                "number_estimation_factor",
            },
        )
    )
    ph3.generate_fc2_displacements(
        **_filtered_kwargs(
            fc2_config,
            {
                "distance",
                "is_plusminus",
                "is_diagonal",
                "number_of_snapshots",
                "random_seed",
                "max_distance",
            },
        )
    )

    out_yaml = paths.output / "phono3py_disp.yaml"
    ph3.save(filename=out_yaml)
    write_crystal_structure(
        paths.output / "POSCAR",
        unitcell,
        interface_mode=interface_mode,
        optional_structure_info=structure_info,
    )
    write_crystal_structure(
        paths.output / "SPOSCAR_FC3",
        ph3.supercell,
        interface_mode=interface_mode,
        optional_structure_info=structure_info,
    )
    write_crystal_structure(
        paths.output / "SPOSCAR_FC2",
        ph3.phonon_supercell,
        interface_mode=interface_mode,
        optional_structure_info=structure_info,
    )

    zfill = int(displacement_config.get("zfill_width", 5))
    fc3_entries = _write_ph3_supercells(
        ph3.supercells_with_displacements,
        paths.output,
        prefix="FC3_POSCAR",
        zfill=zfill,
        interface_mode=interface_mode,
        structure_info=structure_info,
    )
    fc2_entries = _write_ph3_supercells(
        ph3.phonon_supercells_with_displacements,
        paths.output,
        prefix="FC2_POSCAR",
        zfill=zfill,
        interface_mode=interface_mode,
        structure_info=structure_info,
    )
    manifest = {
        "fc3": {
            "entries": fc3_entries,
            "n_atoms": len(ph3.supercell),
        },
        "fc2": {
            "entries": fc2_entries,
            "n_atoms": len(ph3.phonon_supercell),
        },
    }
    _write_json(paths.output / "displacement_manifest.json", manifest)
    _write_json(
        paths.output / "metadata.json",
        {
            "stage": "ph3-displace",
            "input": str(in_poscar),
            "output": str(out_yaml),
            "n_fc3_dataset_entries": len(fc3_entries),
            "n_fc3_calculations": sum(x["filename"] is not None for x in fc3_entries),
            "n_fc2_calculations": sum(x["filename"] is not None for x in fc2_entries),
        },
    )
    _populate_ph3_forces_input(config)


def _run_ph3_forces(config: dict[str, Any], *, force: bool) -> None:
    _populate_ph3_forces_input(config)
    paths = stage_paths(config, "ph3-forces")
    in_yaml = paths.input / "phono3py_disp.yaml"
    if not in_yaml.exists():
        raise FileNotFoundError(
            f"Missing {in_yaml}. Put phono3py_disp.yaml and displaced POSCARs in "
            f"{paths.input}, or run ph3-displace first."
        )

    if _stage_complete(config, "ph3-forces") and not force:
        print(f"[ph3-forces] skip: outputs already exist in {paths.output}")
        _populate_ph3_fc_input(config)
        return

    manifest = _read_or_infer_ph3_manifest(paths.input)
    if force:
        _clear_ph3_force_checkpoints(paths.output)
    for name in (
        "forces_fc3.npy",
        "forces_fc2.npy",
        "energies_fc3.npy",
        "energies_fc2.npy",
        "completed.json",
    ):
        output_file = paths.output / name
        if output_file.exists():
            output_file.unlink()

    calculator = build_calculator(config)
    force_config = config["phono3py"].get("forces", {})
    results: dict[str, tuple[Any, Any]] = {}
    for kind in ("fc3", "fc2"):
        results[kind] = _evaluate_ph3_force_kind(
            kind,
            manifest[kind],
            paths,
            calculator,
            subtract_drift=bool(force_config.get("subtract_drift", True)),
        )

    import numpy as np

    for kind, (force_array, energy_array) in results.items():
        _atomic_save_npy(paths.output / f"forces_{kind}.npy", force_array)
        if bool(force_config.get("write_energies", True)):
            _atomic_save_npy(paths.output / f"energies_{kind}.npy", energy_array)

    completed = {
        "stage": "ph3-forces",
        "input": str(in_yaml),
        "n_fc3_dataset_entries": int(results["fc3"][0].shape[0]),
        "n_fc2_dataset_entries": int(results["fc2"][0].shape[0]),
    }
    _write_json(paths.output / "metadata.json", completed)
    _write_json(paths.output / "completed.json", completed)
    _populate_ph3_fc_input(config)


def _run_ph3_fc(config: dict[str, Any], *, force: bool) -> None:
    _populate_ph3_fc_input(config)
    paths = stage_paths(config, "ph3-fc")
    in_yaml = paths.input / "phono3py_disp.yaml"
    in_fc3_forces = paths.input / "forces_fc3.npy"
    in_fc2_forces = paths.input / "forces_fc2.npy"
    in_forces_fc3 = paths.input / "FORCES_FC3"
    in_forces_fc2 = paths.input / "FORCES_FC2"
    if not in_yaml.exists():
        raise FileNotFoundError(f"Missing {in_yaml}.")
    if not in_fc3_forces.exists() and not in_forces_fc3.exists():
        raise FileNotFoundError(
            f"Missing {in_fc3_forces} or standard phono3py file {in_forces_fc3}."
        )
    if not in_fc2_forces.exists() and not in_forces_fc2.exists():
        raise FileNotFoundError(
            f"Missing {in_fc2_forces} or standard phono3py file {in_forces_fc2}."
        )

    if _stage_complete(config, "ph3-fc") and not force:
        print(f"[ph3-fc] skip: outputs already exist in {paths.output}")
        _populate_kappa_inputs(config)
        return

    import numpy as np
    import phono3py
    from phono3py.file_IO import (
        write_FORCES_FC2,
        write_FORCES_FC3,
        write_fc2_to_hdf5,
        write_fc3_to_hdf5,
    )

    ph3_config = config["phono3py"]
    fc_config = ph3_config.get("force_constants", {})
    ph3 = phono3py.load(
        phono3py_yaml=in_yaml,
        forces_fc3_filename=(
            in_forces_fc3
            if in_forces_fc3.exists() and not in_fc3_forces.exists()
            else None
        ),
        forces_fc2_filename=(
            in_forces_fc2
            if in_forces_fc2.exists() and not in_fc2_forces.exists()
            else None
        ),
        calculator=ph3_config.get("calculator", "vasp"),
        produce_fc=False,
        is_symmetry=bool(ph3_config.get("is_symmetry", True)),
        symprec=float(ph3_config.get("symprec", 1.0e-5)),
        log_level=int(ph3_config.get("log_level", 1)),
        lang=str(ph3_config.get("lang", "Rust")),
    )
    if in_fc3_forces.exists():
        ph3.forces = np.load(in_fc3_forces)
    if in_fc2_forces.exists():
        ph3.phonon_forces = np.load(in_fc2_forces)
    if ph3.forces is None or ph3.phonon_forces is None:
        raise RuntimeError("Could not read both fc3 and fc2 forces.")

    write_FORCES_FC3(
        ph3.dataset,
        forces_fc3=ph3.forces,
        filename=paths.output / "FORCES_FC3",
    )
    write_FORCES_FC2(
        ph3.phonon_dataset,
        forces_fc2=ph3.phonon_forces,
        filename=paths.output / "FORCES_FC2",
    )

    fc_calculator = str(fc_config.get("fc_calculator", "traditional")).strip() or None
    fc_options = str(fc_config.get("fc_calculator_options", "")).strip() or None
    common_kwargs = {
        "is_compact_fc": bool(fc_config.get("is_compact_fc", True)),
        "fc_calculator": fc_calculator,
        "fc_calculator_options": fc_options,
        "use_symfc_projector": bool(fc_config.get("use_symfc_projector", False)),
    }
    ph3.produce_fc3(
        symmetrize_fc3r=bool(fc_config.get("symmetrize_fc3", True)),
        **common_kwargs,
    )
    ph3.produce_fc2(
        symmetrize_fc2=bool(fc_config.get("symmetrize_fc2", True)),
        **common_kwargs,
    )
    if ph3.fc3 is None or ph3.fc2 is None:
        raise RuntimeError("phono3py did not produce both fc3 and fc2.")

    compression = fc_config.get("compression", "gzip")
    write_fc3_to_hdf5(
        ph3.fc3,
        fc3_nonzero_indices=ph3.fc3_nonzero_indices,
        filename=str(paths.output / "fc3.hdf5"),
        p2s_map=ph3.primitive.p2s_map,
        fc3_cutoff=ph3.fc3_cutoff,
        compression=compression,
    )
    write_fc2_to_hdf5(
        ph3.fc2,
        filename=str(paths.output / "fc2.hdf5"),
        p2s_map=ph3.phonon_primitive.p2s_map,
        physical_unit="eV/angstrom^2",
        cutoff=ph3.fc2_cutoff,
        compression=compression,
    )
    ph3.save(filename=paths.output / "phono3py_params.yaml")

    completed = {
        "stage": "ph3-fc",
        "input_phono3py_yaml": str(in_yaml),
        "fc2_shape": list(ph3.fc2.shape),
        "fc3_shape": list(ph3.fc3.shape),
        "fc_calculator": fc_calculator,
    }
    _write_json(paths.output / "metadata.json", completed)
    _write_json(paths.output / "completed.json", completed)
    _populate_kappa_inputs(config)


def _run_thermal_conductivity(
    config: dict[str, Any], *, method: str, force: bool
) -> None:
    stage = f"kappa-{method}"
    _populate_kappa_inputs(config)
    paths = stage_paths(config, stage)
    in_yaml = paths.input / "phono3py_disp.yaml"
    in_fc2 = paths.input / "fc2.hdf5"
    in_fc3 = paths.input / "fc3.hdf5"
    for required in (in_yaml, in_fc2, in_fc3):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Put phono3py_disp.yaml, fc2.hdf5, and "
                f"fc3.hdf5 in {paths.input}, or run ph3-fc first."
            )

    if _stage_complete(config, stage) and not force:
        print(f"[{stage}] skip: {paths.output / 'completed.json'} exists")
        return
    completed_file = paths.output / "completed.json"
    if completed_file.exists():
        completed_file.unlink()

    import numpy as np
    import phono3py

    ph3_config = config["phono3py"]
    thermal_config = config["thermal_conductivity"]
    born_file = paths.input / "BORN"
    ph3 = phono3py.load(
        phono3py_yaml=in_yaml,
        fc2_filename=in_fc2,
        fc3_filename=in_fc3,
        born_filename=born_file if born_file.exists() else None,
        calculator=ph3_config.get("calculator", "vasp"),
        produce_fc=False,
        is_symmetry=bool(ph3_config.get("is_symmetry", True)),
        is_mesh_symmetry=bool(ph3_config.get("is_mesh_symmetry", True)),
        use_grg=bool(ph3_config.get("use_grg", False)),
        make_r0_average=bool(ph3_config.get("make_r0_average", True)),
        symprec=float(ph3_config.get("symprec", 1.0e-5)),
        log_level=int(ph3_config.get("log_level", 1)),
        lang=str(ph3_config.get("lang", "Rust")),
    )
    ph3.mesh_numbers = thermal_config.get("mesh", [8, 8, 8])
    sigmas = thermal_config.get("sigmas", [])
    ph3.sigmas = (
        [None]
        if not sigmas
        else [None if x == "tetrahedron" else float(x) for x in sigmas]
    )
    if thermal_config.get("sigma_cutoff") is not None:
        ph3.sigma_cutoff = float(thermal_config["sigma_cutoff"])
    if thermal_config.get("band_indices"):
        ph3.band_indices = thermal_config["band_indices"]

    ph3.init_phph_interaction(
        nac_q_direction=thermal_config.get("nac_q_direction") or None,
        constant_averaged_interaction=_optional_float(
            thermal_config.get("constant_averaged_interaction")
        ),
        frequency_scale_factor=_optional_float(
            thermal_config.get("frequency_scale_factor")
        ),
        symmetrize_fc3q=bool(thermal_config.get("symmetrize_fc3q", False)),
        lapack_zheev_uplo=thermal_config.get("lapack_zheev_uplo"),
        openmp_per_triplets=thermal_config.get("openmp_per_triplets"),
    )

    temperatures = thermal_config.get("temperatures") or None
    common_kwargs = {
        "temperatures": temperatures,
        "is_isotope": bool(thermal_config.get("is_isotope", False)),
        "mass_variances": thermal_config.get("mass_variances") or None,
        "grid_points": thermal_config.get("grid_points") or None,
        "boundary_mfp": _optional_float(thermal_config.get("boundary_mfp")),
        "solve_collective_phonon": bool(
            thermal_config.get("solve_collective_phonon", False)
        ),
        "is_kappa_star": bool(thermal_config.get("is_kappa_star", True)),
        "gv_delta_q": _optional_float(thermal_config.get("gv_delta_q")),
        "is_full_pp": bool(thermal_config.get("is_full_pp", False)),
        "transport_type": thermal_config.get("transport_type"),
        "write_kappa": True,
        "compression": thermal_config.get("compression", "gzip"),
        "log_level": int(ph3_config.get("log_level", 1)),
    }
    if method == "rta":
        method_config = thermal_config.get("rta", {})
        method_kwargs = {
            "use_ave_pp": bool(method_config.get("use_ave_pp", False)),
            "write_gamma": bool(method_config.get("write_gamma", False)),
            "read_gamma": bool(method_config.get("read_gamma", False)),
            "is_N_U": bool(method_config.get("is_N_U", False)),
            "write_gamma_detail": bool(method_config.get("write_gamma_detail", False)),
            "write_pp": bool(method_config.get("write_pp", False)),
            "read_pp": bool(method_config.get("read_pp", False)),
        }
    else:
        method_config = thermal_config.get("iterative", {})
        read_collision = method_config.get("read_collision", False)
        method_kwargs = {
            "is_reducible_collision_matrix": bool(
                method_config.get("is_reducible_collision_matrix", False)
            ),
            "pinv_cutoff": _optional_float(method_config.get("pinv_cutoff")),
            "pinv_method": int(method_config.get("pinv_method", 0)),
            "pinv_solver": int(method_config.get("pinv_solver", 0)),
            "write_collision": bool(method_config.get("write_collision", False)),
            "read_collision": read_collision if read_collision else None,
            "write_pp": bool(method_config.get("write_pp", False)),
            "read_pp": bool(method_config.get("read_pp", False)),
            "write_LBTE_solution": bool(method_config.get("write_LBTE_solution", False)),
        }

    with _working_directory(paths.output):
        ph3.run_thermal_conductivity(
            is_LBTE=method == "iterative",
            **common_kwargs,
            **method_kwargs,
        )

    result = ph3.thermal_conductivity
    result_arrays = {}
    for name in ("temperatures", "kappa", "kappa_RTA", "mode_kappa", "mode_kappa_RTA"):
        value = getattr(result, name, None)
        if value is not None:
            result_arrays[name] = np.asarray(value)
    if result_arrays:
        np.savez_compressed(paths.output / "thermal_conductivity.npz", **result_arrays)

    generated_hdf5 = sorted(path.name for path in paths.output.glob("*.hdf5"))
    completed = {
        "stage": stage,
        "method": "RTA" if method == "rta" else "full iterative LBTE",
        "mesh": thermal_config.get("mesh", [8, 8, 8]),
        "temperatures": temperatures,
        "hdf5_outputs": generated_hdf5,
    }
    _write_json(paths.output / "metadata.json", completed)
    _write_json(completed_file, completed)


def _band_nqpoints(band_config: dict[str, Any]) -> int:
    if "nqpoints" in band_config:
        value = band_config["nqpoints"]
    else:
        value = band_config.get("npoints", 101)
    nqpoints = int(value)
    if nqpoints < 2:
        raise ValueError("band.nqpoints must be >= 2.")
    return nqpoints


def _populate_displace_input(config: dict[str, Any]) -> None:
    relax_out = stage_paths(config, "relax").output / "POSCAR"
    displace_in = stage_paths(config, "displace").input / "POSCAR"
    if relax_out.exists():
        _copy_file(relax_out, displace_in, overwrite=_overwrite_next_inputs(config))


def _populate_ph3_displace_input(config: dict[str, Any]) -> None:
    relax_out = stage_paths(config, "relax").output / "POSCAR"
    ph3_in = stage_paths(config, "ph3-displace").input / "POSCAR"
    if relax_out.exists():
        _copy_file(relax_out, ph3_in, overwrite=_overwrite_next_inputs(config))


def _populate_forces_input(config: dict[str, Any]) -> None:
    displace_paths = stage_paths(config, "displace")
    forces_in = stage_paths(config, "forces").input
    overwrite = _overwrite_next_inputs(config)
    for name in ("POSCAR", "SPOSCAR", "phonopy_disp.yaml"):
        src = displace_paths.output / name
        if src.exists():
            _copy_file(src, forces_in / name, overwrite=overwrite)
    for src in _displacement_poscars(displace_paths.output):
        _copy_file(src, forces_in / src.name, overwrite=overwrite)


def _populate_band_input(config: dict[str, Any]) -> None:
    forces_out = stage_paths(config, "forces").output
    forces_in = stage_paths(config, "forces").input
    displace_in = stage_paths(config, "displace").input
    band_in = stage_paths(config, "band").input
    overwrite = _overwrite_next_inputs(config)

    for src in (forces_out / "FORCE_SETS", forces_in / "phonopy_disp.yaml", displace_in / "POSCAR"):
        if src.exists():
            _copy_file(src, band_in / src.name, overwrite=overwrite)

    params = forces_out / "phonopy_params.yaml"
    if params.exists():
        _copy_file(params, band_in / params.name, overwrite=overwrite)


def _populate_ph3_forces_input(config: dict[str, Any]) -> None:
    source_paths = stage_paths(config, "ph3-displace")
    destination = stage_paths(config, "ph3-forces").input
    overwrite = _overwrite_next_inputs(config)
    fixed_names = (
        "POSCAR",
        "SPOSCAR_FC3",
        "SPOSCAR_FC2",
        "phono3py_disp.yaml",
        "displacement_manifest.json",
    )
    for name in fixed_names:
        source = source_paths.output / name
        if source.exists():
            _copy_file(source, destination / name, overwrite=overwrite)
    for source in _ph3_displacement_poscars(source_paths.output):
        _copy_file(source, destination / source.name, overwrite=overwrite)
    born = source_paths.input / "BORN"
    if born.exists():
        _copy_file(born, destination / "BORN", overwrite=overwrite)


def _populate_ph3_fc_input(config: dict[str, Any]) -> None:
    force_paths = stage_paths(config, "ph3-forces")
    destination = stage_paths(config, "ph3-fc").input
    overwrite = _overwrite_next_inputs(config)
    for source in (
        force_paths.input / "phono3py_disp.yaml",
        force_paths.output / "forces_fc3.npy",
        force_paths.output / "forces_fc2.npy",
        force_paths.output / "energies_fc3.npy",
        force_paths.output / "energies_fc2.npy",
        force_paths.input / "BORN",
    ):
        if source.exists():
            _copy_file(source, destination / source.name, overwrite=overwrite)


def _populate_kappa_inputs(config: dict[str, Any]) -> None:
    fc_paths = stage_paths(config, "ph3-fc")
    overwrite = _overwrite_next_inputs(config)
    sources = (
        fc_paths.input / "phono3py_disp.yaml",
        fc_paths.output / "phono3py_params.yaml",
        fc_paths.output / "fc3.hdf5",
        fc_paths.output / "fc2.hdf5",
        fc_paths.input / "BORN",
    )
    for stage in ("kappa-rta", "kappa-iterative"):
        destination = stage_paths(config, stage).input
        for source in sources:
            if source.exists():
                _copy_file(source, destination / source.name, overwrite=overwrite)


def _overwrite_next_inputs(config: dict[str, Any]) -> bool:
    return bool(config["workflow"].get("overwrite_next_inputs", False))


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _displacement_poscars(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("POSCAR-*")
        if path.is_file() and path.name.split("-")[-1].isdigit()
    )


def _ph3_displacement_poscars(directory: Path) -> list[Path]:
    return sorted(
        path
        for prefix in ("FC3_POSCAR-*", "FC2_POSCAR-*")
        for path in directory.glob(prefix)
        if path.is_file() and path.name.split("-")[-1].isdigit()
    )


def _stage_complete(config: dict[str, Any], stage: str) -> bool:
    output = stage_paths(config, stage).output
    return all((output / name).exists() for name in EXPECTED_OUTPUTS[stage])


def _write_ph3_supercells(
    supercells: Iterable[Any],
    output_dir: Path,
    *,
    prefix: str,
    zfill: int,
    interface_mode: str,
    structure_info: Any,
) -> list[dict[str, Any]]:
    from phonopy.interface.calculator import write_crystal_structure

    entries = []
    for index, supercell in enumerate(supercells, start=1):
        filename = None
        if supercell is not None:
            filename = f"{prefix}-{index:0{zfill}d}"
            write_crystal_structure(
                output_dir / filename,
                supercell,
                interface_mode=interface_mode,
                optional_structure_info=structure_info,
            )
        entries.append({"index": index, "filename": filename})
    return entries


def _clear_ph3_displacement_outputs(output_dir: Path) -> None:
    for path in _ph3_displacement_poscars(output_dir):
        path.unlink()
    for name in (
        "POSCAR",
        "SPOSCAR_FC3",
        "SPOSCAR_FC2",
        "phono3py_disp.yaml",
        "displacement_manifest.json",
        "metadata.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _read_or_infer_ph3_manifest(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "displacement_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    from ase.io import read

    manifest = {}
    for kind, pattern in (("fc3", "FC3_POSCAR-*"), ("fc2", "FC2_POSCAR-*")):
        files = sorted(
            path
            for path in input_dir.glob(pattern)
            if path.is_file() and path.name.split("-")[-1].isdigit()
        )
        if not files:
            raise FileNotFoundError(
                f"No {pattern} files found in {input_dir}. A manually supplied "
                "ph3-forces input needs both FC3_POSCAR-* and FC2_POSCAR-* files."
            )
        indices = [int(path.name.split("-")[-1]) for path in files]
        files_by_index = dict(zip(indices, files))
        entries = [
            {
                "index": index,
                "filename": files_by_index[index].name if index in files_by_index else None,
            }
            for index in range(1, max(indices) + 1)
        ]
        manifest[kind] = {
            "entries": entries,
            "n_atoms": len(read(files[0], format="vasp")),
        }
    return manifest


def _evaluate_ph3_force_kind(
    kind: str,
    manifest: dict[str, Any],
    paths: StagePaths,
    calculator: Any,
    *,
    subtract_drift: bool,
):
    import numpy as np
    from ase.io import read

    entries = manifest["entries"]
    n_atoms = int(manifest["n_atoms"])
    checkpoint_dir = paths.output / "checkpoints" / kind
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    forces = []
    energies = []
    n_calculations = sum(entry.get("filename") is not None for entry in entries)
    calculation_index = 0

    for entry in entries:
        index = int(entry["index"])
        filename = entry.get("filename")
        if filename is None:
            forces.append(np.zeros((n_atoms, 3), dtype=float))
            energies.append(np.nan)
            continue

        calculation_index += 1
        checkpoint = checkpoint_dir / f"force-{index:05d}.npz"
        if checkpoint.exists():
            with np.load(checkpoint) as data:
                force = np.asarray(data["forces"], dtype=float)
                energy = float(data["energy"])
            if force.shape != (n_atoms, 3):
                raise RuntimeError(
                    f"Invalid checkpoint shape in {checkpoint}: {force.shape}, "
                    f"expected {(n_atoms, 3)}."
                )
            print(
                f"[ph3-forces:{kind}] resume "
                f"{calculation_index:05d}/{n_calculations:05d}"
            )
        else:
            source = paths.input / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing displaced structure: {source}")
            atoms = read(source, format="vasp")
            if len(atoms) != n_atoms:
                raise RuntimeError(
                    f"{source} has {len(atoms)} atoms, but manifest expects {n_atoms}."
                )
            atoms.calc = calculator
            force = np.asarray(atoms.get_forces(), dtype=float)
            drift = force.mean(axis=0)
            if subtract_drift:
                force = force - drift[None, :]
            energy_value = _safe_float(lambda: atoms.get_potential_energy())
            energy = np.nan if energy_value is None else energy_value
            _atomic_save_npz(checkpoint, forces=force, energy=np.asarray(energy))
            print(
                f"[ph3-forces:{kind}] {calculation_index:05d}/{n_calculations:05d} "
                f"{filename}, drift={np.linalg.norm(drift):.3e} eV/A"
            )
        forces.append(force)
        energies.append(energy)

    return np.asarray(forces, dtype=float), np.asarray(energies, dtype=float)


def _clear_ph3_force_checkpoints(output_dir: Path) -> None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.exists():
        return
    for path in checkpoint_root.glob("*/*.npz"):
        path.unlink()


def _atomic_save_npy(path: Path, array: Any) -> None:
    import numpy as np

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _optional_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _phonopy_atoms_to_ase(cell: Any):
    import numpy as np
    from ase import Atoms

    return Atoms(
        symbols=list(cell.symbols),
        cell=np.asarray(cell.cell, dtype=float),
        scaled_positions=np.asarray(cell.scaled_positions, dtype=float),
        pbc=True,
    )


def _filtered_kwargs(source: dict[str, Any], allowed: Iterable[str]) -> dict[str, Any]:
    allowed_set = set(allowed)
    return {key: value for key, value in source.items() if key in allowed_set}


def _clear_relax_success_outputs(paths: StagePaths) -> None:
    for name in ("POSCAR", "CONTCAR", "relaxed.extxyz"):
        path = paths.output / name
        if path.exists():
            path.unlink()


def _clear_matching_displace_input(config: dict[str, Any], relax_poscar: Path) -> None:
    displace_poscar = stage_paths(config, "displace").input / "POSCAR"
    if (
        relax_poscar.exists()
        and displace_poscar.exists()
        and filecmp.cmp(relax_poscar, displace_poscar, shallow=False)
    ):
        displace_poscar.unlink()


def _clear_matching_ph3_displace_input(
    config: dict[str, Any], relax_poscar: Path
) -> None:
    ph3_poscar = stage_paths(config, "ph3-displace").input / "POSCAR"
    if (
        relax_poscar.exists()
        and ph3_poscar.exists()
        and filecmp.cmp(relax_poscar, ph3_poscar, shallow=False)
    ):
        ph3_poscar.unlink()


def _max_force(atoms: Any) -> float:
    forces = atoms.get_forces()
    if len(forces) == 0:
        return 0.0
    return max(
        float(sum(component * component for component in force) ** 0.5)
        for force in forces
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(func):
    try:
        value = func()
    except Exception:
        return None
    if value is None:
        return None
    return float(value)
