from __future__ import annotations

from typing import Any

from .calculators import build_calculator
from .stage_archive import (
    _archive_dos_inputs,
    _archive_kappa_outputs,
    _plot_kappa_outputs,
    plot_dos_archive,
)
from .stage_common import (
    _filtered_kwargs,
    _populate_kappa_inputs,
    _populate_ph3_displace_input,
    _populate_ph3_fc_input,
    _populate_ph3_forces_input,
    _stage_complete,
    _write_json,
    stage_paths,
)
from .phono3py_utils import (
    _atomic_save_npy,
    _clear_ph3_displacement_outputs,
    _clear_ph3_force_checkpoints,
    _evaluate_ph3_force_kind,
    _optional_float,
    _read_or_infer_ph3_manifest,
    _working_directory,
    _write_ph3_supercells,
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
    if _stage_complete(config, "ph3-fc") and not force:
        print(f"[ph3-fc] skip: outputs already exist in {paths.output}")
        dos_archive_outputs = _archive_dos_inputs(config)
        if dos_archive_outputs and bool(config.get("dos", {}).get("enabled", True)):
            plot_dos_archive(config)
        _populate_kappa_inputs(config)
        return

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

    dos_archive_outputs = _archive_dos_inputs(config)
    dos_plot_outputs = None
    if dos_archive_outputs and bool(config.get("dos", {}).get("enabled", True)):
        dos_plot_outputs = plot_dos_archive(config)

    completed = {
        "stage": "ph3-fc",
        "input_phono3py_yaml": str(in_yaml),
        "fc2_shape": list(ph3.fc2.shape),
        "fc3_shape": list(ph3.fc3.shape),
        "fc_calculator": fc_calculator,
        "dos_archive_inputs": dos_archive_outputs,
        "dos_plot_outputs": dos_plot_outputs,
    }
    _write_json(paths.output / "metadata.json", completed)
    _write_json(paths.output / "completed.json", completed)
    _populate_kappa_inputs(config)


def _run_thermal_conductivity(
    config: dict[str, Any], *, method: str, force: bool
) -> None:
    stage = f"kappa-{method}"
    plot_method = "rta" if method == "rta" else "lbte"
    _populate_kappa_inputs(config)
    paths = stage_paths(config, stage)
    if _stage_complete(config, stage) and not force:
        print(f"[{stage}] skip: {paths.output / 'completed.json'} exists")
        _plot_kappa_outputs(paths.output, config, method=plot_method)
        _archive_kappa_outputs(config, paths.output, method=plot_method)
        return

    in_yaml = paths.input / "phono3py_disp.yaml"
    in_fc2 = paths.input / "fc2.hdf5"
    in_fc3 = paths.input / "fc3.hdf5"
    for required in (in_yaml, in_fc2, in_fc3):
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Put phono3py_disp.yaml, fc2.hdf5, and "
                f"fc3.hdf5 in {paths.input}, or run ph3-fc first."
            )

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
    plot_outputs = _plot_kappa_outputs(paths.output, config, method=plot_method)
    archive_outputs = _archive_kappa_outputs(
        config, paths.output, method=plot_method
    )
    completed = {
        "stage": stage,
        "method": "RTA" if method == "rta" else "full iterative LBTE",
        "mesh": thermal_config.get("mesh", [8, 8, 8]),
        "temperatures": temperatures,
        "hdf5_outputs": generated_hdf5,
        "plot_outputs": plot_outputs,
        "archive_outputs": archive_outputs,
    }
    _write_json(paths.output / "metadata.json", completed)
    _write_json(completed_file, completed)
