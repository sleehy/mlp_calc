from __future__ import annotations

from typing import Any

from .calculators import build_calculator
from .config import input_poscar
from .stage_archive import _archive_band_outputs
from .stage_common import (
    _band_nqpoints,
    _clear_matching_displace_input,
    _clear_matching_ph3_displace_input,
    _clear_relax_success_outputs,
    _copy_file,
    _displacement_poscars,
    _filtered_kwargs,
    _max_force,
    _phonopy_atoms_to_ase,
    _populate_band_input,
    _populate_displace_input,
    _populate_forces_input,
    _populate_ph3_displace_input,
    _safe_float,
    _write_json,
    stage_paths,
)


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
    out_band = paths.output / "band.yaml"
    if out_band.exists() and not force:
        print(f"[band] skip: {out_band} exists")
        _archive_band_outputs(config, paths.output)
        return

    in_yaml = paths.input / "phonopy_disp.yaml"
    in_force_sets = paths.input / "FORCE_SETS"
    if not in_yaml.exists():
        raise FileNotFoundError(f"Missing {in_yaml}.")
    if not in_force_sets.exists():
        raise FileNotFoundError(f"Missing {in_force_sets}.")

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
    archive_outputs = _archive_band_outputs(config, paths.output)
    _write_json(
        paths.output / "metadata.json",
        {
            "stage": "band",
            "input_force_sets": str(in_force_sets),
            "input_phonopy_yaml": str(in_yaml),
            "output": str(out_band),
            "archive_outputs": archive_outputs,
        },
    )

