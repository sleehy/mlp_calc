from __future__ import annotations

import filecmp
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .calculators import build_calculator
from .config import input_poscar, run_dir


STAGES = ("relax", "displace", "forces", "band")

STAGE_DIR_NAMES = {
    "relax": "01_relax",
    "displace": "02_displacements",
    "forces": "03_forces",
    "band": "04_band",
}

EXPECTED_OUTPUTS = {
    "relax": "POSCAR",
    "displace": "phonopy_disp.yaml",
    "forces": "FORCE_SETS",
    "band": "band.yaml",
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


def stage_status(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for stage in STAGES:
        paths = stage_paths(config, stage)
        expected = paths.output / EXPECTED_OUTPUTS[stage]
        rows.append(
            {
                "stage": stage,
                "input": str(paths.input),
                "output": str(expected),
                "status": "done" if expected.exists() else "pending",
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
        return

    if force:
        _clear_matching_displace_input(config, out_poscar)
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
