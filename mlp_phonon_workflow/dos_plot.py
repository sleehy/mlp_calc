from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def calculate_dos_from_fc2(
    phono3py_yaml: str | Path,
    fc2_file: str | Path,
    *,
    mesh: Sequence[int],
    calculator: str = "vasp",
    born_file: str | Path | None = None,
    method: str = "tetrahedron",
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
    projected: bool = False,
    symprec: float = 1.0e-5,
) -> dict[str, Any]:
    """Calculate harmonic total DOS and optional element-projected DOS."""
    import phono3py
    from phonopy import Phonopy

    mesh_numbers = _validate_mesh(mesh)
    method, effective_sigma = _validate_dos_settings(
        method, sigma, frequency_pitch
    )

    yaml_source = Path(phono3py_yaml).expanduser().resolve()
    fc2_source = Path(fc2_file).expanduser().resolve()
    for source in (yaml_source, fc2_source):
        if not source.exists():
            raise FileNotFoundError(f"DOS input not found: {source}")
    born_source = (
        Path(born_file).expanduser().resolve() if born_file is not None else None
    )
    if born_source is not None and not born_source.exists():
        raise FileNotFoundError(f"DOS BORN file not found: {born_source}")

    ph3 = phono3py.load(
        phono3py_yaml=yaml_source,
        fc2_filename=fc2_source,
        born_filename=born_source,
        calculator=calculator,
        produce_fc=False,
        is_symmetry=True,
        symprec=float(symprec),
        log_level=0,
    )
    if ph3.fc2 is None:
        raise RuntimeError(f"Could not load second-order force constants: {fc2_source}")

    phonon = Phonopy(
        ph3.unitcell,
        supercell_matrix=ph3.phonon_supercell_matrix,
        primitive_matrix=ph3.primitive_matrix,
        calculator=calculator,
        is_symmetry=True,
        symprec=float(symprec),
    )
    if ph3.nac_params is not None:
        phonon.nac_params = ph3.nac_params
    phonon.force_constants = ph3.fc2

    return _calculate_dos_with_phonopy(
        phonon,
        mesh=mesh_numbers,
        method=method,
        sigma=effective_sigma,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_pitch=frequency_pitch,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        projected=projected,
        source_fields={
            "input_kind": "phono3py_fc2",
            "phono3py_yaml": yaml_source,
            "fc2": fc2_source,
            "phonopy_yaml": None,
            "force_constants": None,
            "born": born_source,
        },
    )


def calculate_dos_from_phonopy(
    phonopy_yaml: str | Path,
    *,
    mesh: Sequence[int],
    force_constants_file: str | Path | None = None,
    calculator: str = "vasp",
    born_file: str | Path | None = None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    method: str = "tetrahedron",
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
    projected: bool = False,
    symprec: float = 1.0e-5,
) -> dict[str, Any]:
    """Calculate DOS from a phonopy YAML and optional force-constants file."""
    import phonopy

    mesh_numbers = _validate_mesh(mesh)
    method, effective_sigma = _validate_dos_settings(
        method, sigma, frequency_pitch
    )
    yaml_source = Path(phonopy_yaml).expanduser().resolve()
    if not yaml_source.exists():
        raise FileNotFoundError(f"DOS input not found: {yaml_source}")
    force_constants_source = (
        Path(force_constants_file).expanduser().resolve()
        if force_constants_file is not None
        else None
    )
    if force_constants_source is not None and not force_constants_source.exists():
        raise FileNotFoundError(
            f"DOS force-constants input not found: {force_constants_source}"
        )
    born_source = (
        Path(born_file).expanduser().resolve() if born_file is not None else None
    )
    if born_source is not None and not born_source.exists():
        raise FileNotFoundError(f"DOS BORN file not found: {born_source}")

    phonon = phonopy.load(
        phonopy_yaml=yaml_source,
        force_constants_filename=force_constants_source,
        born_filename=born_source,
        calculator=calculator,
        produce_fc=True,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        is_symmetry=True,
        symprec=float(symprec),
        log_level=0,
    )
    if phonon.force_constants is None:
        raise RuntimeError(
            "Could not load or reconstruct force constants from "
            f"{yaml_source}. Provide a phonopy_params.yaml containing forces "
            "or use --force-constants."
        )

    return _calculate_dos_with_phonopy(
        phonon,
        mesh=mesh_numbers,
        method=method,
        sigma=effective_sigma,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_pitch=frequency_pitch,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        projected=projected,
        source_fields={
            "input_kind": "phonopy",
            "phono3py_yaml": None,
            "fc2": None,
            "phonopy_yaml": yaml_source,
            "force_constants": force_constants_source,
            "born": born_source,
        },
    )


def _calculate_dos_with_phonopy(
    phonon: Any,
    *,
    mesh: Sequence[int],
    method: str,
    sigma: float | None,
    frequency_min: float | None,
    frequency_max: float | None,
    frequency_pitch: float | None,
    is_gamma_center: bool,
    is_mesh_symmetry: bool,
    projected: bool,
    source_fields: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    # Phonopy requires eigenvectors on the full mesh for projected DOS.
    effective_mesh_symmetry = bool(is_mesh_symmetry) and not bool(projected)
    phonon.run_mesh(
        mesh,
        is_mesh_symmetry=effective_mesh_symmetry,
        is_gamma_center=bool(is_gamma_center),
        with_eigenvectors=bool(projected),
    )
    phonon.run_total_dos(
        sigma=sigma,
        freq_min=frequency_min,
        freq_max=frequency_max,
        freq_pitch=frequency_pitch,
        use_tetrahedron_method=method == "tetrahedron",
    )
    if phonon.total_dos is None:
        raise RuntimeError("Phonopy did not produce a total DOS result.")

    result = {
        "frequency": np.asarray(phonon.total_dos.frequency_points, dtype=float),
        "dos": np.asarray(phonon.total_dos.dos, dtype=float),
        "projected_dos": None,
        "mesh": list(mesh),
        "method": method,
        "sigma": sigma,
        "is_mesh_symmetry": effective_mesh_symmetry,
        **source_fields,
    }
    if projected:
        phonon.run_projected_dos(
            sigma=sigma,
            freq_min=frequency_min,
            freq_max=frequency_max,
            freq_pitch=frequency_pitch,
            use_tetrahedron_method=method == "tetrahedron",
        )
        if phonon.projected_dos is None:
            raise RuntimeError("Phonopy did not produce a projected DOS result.")

        projected_frequency = np.asarray(
            phonon.projected_dos.frequency_points, dtype=float
        )
        if (
            projected_frequency.shape != result["frequency"].shape
            or not np.allclose(projected_frequency, result["frequency"])
        ):
            raise RuntimeError(
                "Total and projected DOS were calculated on different frequency grids."
            )

        atom_projected = np.asarray(
            phonon.projected_dos.projected_dos, dtype=float
        )
        symbols = list(phonon.primitive.symbols)
        if atom_projected.shape[0] != len(symbols):
            raise RuntimeError(
                "Projected DOS atom count does not match the primitive cell."
            )
        element_projected: dict[str, Any] = {}
        for symbol, atom_dos in zip(symbols, atom_projected, strict=True):
            if symbol not in element_projected:
                element_projected[symbol] = np.zeros_like(atom_dos)
            element_projected[symbol] += atom_dos
        result["projected_dos"] = element_projected

    return result


def _validate_dos_settings(
    method: str,
    sigma: float | None,
    frequency_pitch: float | None,
) -> tuple[str, float | None]:
    method = str(method).lower()
    if method not in {"tetrahedron", "gaussian"}:
        raise ValueError("DOS method must be 'tetrahedron' or 'gaussian'.")
    if method == "gaussian" and (sigma is None or float(sigma) <= 0):
        raise ValueError("Gaussian DOS requires a positive sigma.")
    if frequency_pitch is not None and float(frequency_pitch) <= 0:
        raise ValueError("DOS frequency_pitch must be positive.")
    return method, float(sigma) if method == "gaussian" else None


def plot_dos_from_fc2(
    phono3py_yaml: str | Path,
    fc2_file: str | Path,
    *,
    mesh: Sequence[int],
    output_dir: str | Path,
    source_label: str,
    calculator: str = "vasp",
    born_file: str | Path | None = None,
    method: str = "tetrahedron",
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
    projected: bool = False,
    symprec: float = 1.0e-5,
    dpi: int = 200,
) -> dict[str, Any]:
    """Calculate and plot harmonic total and optional projected DOS."""
    result = calculate_dos_from_fc2(
        phono3py_yaml,
        fc2_file,
        mesh=mesh,
        calculator=calculator,
        born_file=born_file,
        method=method,
        sigma=sigma,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_pitch=frequency_pitch,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        projected=projected,
        symprec=symprec,
    )
    return _write_dos_outputs(
        result,
        output_dir=output_dir,
        source_label=source_label,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        dpi=dpi,
    )


def plot_dos_from_phonopy(
    phonopy_yaml: str | Path,
    *,
    mesh: Sequence[int],
    output_dir: str | Path,
    source_label: str,
    force_constants_file: str | Path | None = None,
    calculator: str = "vasp",
    born_file: str | Path | None = None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    method: str = "tetrahedron",
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    is_gamma_center: bool = True,
    is_mesh_symmetry: bool = True,
    projected: bool = False,
    symprec: float = 1.0e-5,
    dpi: int = 200,
) -> dict[str, Any]:
    """Calculate and plot DOS from a harmonic phonopy result."""
    result = calculate_dos_from_phonopy(
        phonopy_yaml,
        mesh=mesh,
        force_constants_file=force_constants_file,
        calculator=calculator,
        born_file=born_file,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        method=method,
        sigma=sigma,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_pitch=frequency_pitch,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        projected=projected,
        symprec=symprec,
    )
    return _write_dos_outputs(
        result,
        output_dir=output_dir,
        source_label=source_label,
        is_gamma_center=is_gamma_center,
        is_mesh_symmetry=is_mesh_symmetry,
        dpi=dpi,
    )


def _build_band_with_projected_dos_from_phonopy(
    phonopy_yaml: str | Path,
    *,
    mesh: Sequence[int],
    force_constants_file: str | Path | None = None,
    calculator: str = "vasp",
    born_file: str | Path | None = None,
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    band_auto: bool = True,
    band_nqpoints: int = 101,
    band_paths: Sequence[Sequence[Sequence[float]]] | None = None,
    band_labels: Sequence[str] | None = None,
    is_band_connection: bool = False,
    method: str = "tetrahedron",
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    is_gamma_center: bool = True,
    symprec: float = 1.0e-5,
) -> tuple["Figure", dict[str, Any]]:
    """Build a Phonopy band+PDOS figure and its metadata without saving."""
    import phonopy

    from phonopy.phonon.band_structure import get_band_qpoints

    mesh_numbers = _validate_mesh(mesh)
    method, effective_sigma = _validate_dos_settings(
        method, sigma, frequency_pitch
    )
    yaml_source = Path(phonopy_yaml).expanduser().resolve()
    if not yaml_source.exists():
        raise FileNotFoundError(f"Band+PDOS input not found: {yaml_source}")
    force_constants_source = (
        Path(force_constants_file).expanduser().resolve()
        if force_constants_file is not None
        else None
    )
    if force_constants_source is not None and not force_constants_source.exists():
        raise FileNotFoundError(
            f"Band+PDOS force-constants input not found: {force_constants_source}"
        )
    born_source = (
        Path(born_file).expanduser().resolve() if born_file is not None else None
    )
    if born_source is not None and not born_source.exists():
        raise FileNotFoundError(f"Band+PDOS BORN file not found: {born_source}")

    phonon = phonopy.load(
        phonopy_yaml=yaml_source,
        force_constants_filename=force_constants_source,
        born_filename=born_source,
        calculator=calculator,
        produce_fc=True,
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        is_symmetry=True,
        symprec=float(symprec),
        log_level=0,
    )
    if phonon.force_constants is None:
        raise RuntimeError(
            "Could not load or reconstruct force constants from "
            f"{yaml_source}. Provide forces in the YAML or use "
            "--force-constants."
        )

    if bool(band_auto):
        phonon.auto_band_structure(
            npoints=int(band_nqpoints),
            with_eigenvectors=False,
            with_group_velocities=False,
            plot=False,
            write_yaml=False,
        )
    else:
        if not band_paths:
            raise ValueError("band.auto=false requires band.paths.")
        qpoint_paths = get_band_qpoints(band_paths, int(band_nqpoints))
        phonon.run_band_structure(
            qpoint_paths,
            with_eigenvectors=False,
            with_group_velocities=False,
            is_band_connection=bool(is_band_connection),
            labels=list(band_labels) if band_labels else None,
        )

    phonon.run_mesh(
        mesh_numbers,
        is_mesh_symmetry=False,
        is_gamma_center=bool(is_gamma_center),
        with_eigenvectors=True,
    )
    phonon.run_projected_dos(
        sigma=effective_sigma,
        freq_min=frequency_min,
        freq_max=frequency_max,
        freq_pitch=frequency_pitch,
        use_tetrahedron_method=method == "tetrahedron",
    )

    element_indices: dict[str, list[int]] = {}
    for index, symbol in enumerate(phonon.primitive.symbols):
        element_indices.setdefault(symbol, []).append(index)
    pdos_indices = list(element_indices.values())

    plot_module = phonon.plot_band_structure_and_dos(pdos_indices=pdos_indices)
    figure = plot_module.gcf()
    figure.set_size_inches(10.0, 5.8)
    dos_candidates = [
        axis
        for axis in figure.axes
        if len(axis.lines) == len(element_indices) + 1
    ]
    if not dos_candidates:
        raise RuntimeError(
            "Could not identify the projected-DOS axis created by Phonopy."
        )
    dos_axis = dos_candidates[-1]
    dos_axis.set_gid("projected_dos")
    for line, symbol in zip(
        dos_axis.lines[: len(element_indices)],
        element_indices,
        strict=True,
    ):
        line.set_label(symbol)
    dos_axis.legend(frameon=False, fontsize=9)
    dos_axis.set_xlabel("Projected DOS")
    populated_axes = [axis for axis in figure.axes if axis.lines]
    band_axes = [axis for axis in populated_axes if axis is not dos_axis]
    for index, axis in enumerate(band_axes):
        axis.set_gid(f"band_{index}")
    figure.suptitle(
        "Phonon band structure and element-projected DOS",
        fontsize=15,
    )

    metadata = {
        "phonopy_yaml": str(yaml_source),
        "force_constants": (
            str(force_constants_source)
            if force_constants_source is not None
            else None
        ),
        "born": str(born_source) if born_source is not None else None,
        "mesh": mesh_numbers,
        "method": method,
        "sigma_THz": effective_sigma,
        "is_gamma_center": bool(is_gamma_center),
        "band_auto": bool(band_auto),
        "band_nqpoints": int(band_nqpoints),
        "projected_by": "element",
        "projected_elements": list(element_indices),
        "projected_atom_indices_zero_based": element_indices,
        "plot_backend": "phonopy.plot_band_structure_and_dos",
    }
    return figure, metadata


def plot_band_with_projected_dos_from_phonopy(
    phonopy_yaml: str | Path,
    *,
    mesh: Sequence[int],
    **kwargs: Any,
) -> "Figure":
    """Return a Matplotlib Figure for external customization without saving."""
    figure, _ = _build_band_with_projected_dos_from_phonopy(
        phonopy_yaml,
        mesh=mesh,
        **kwargs,
    )
    return figure


def save_band_with_projected_dos_from_phonopy(
    phonopy_yaml: str | Path,
    *,
    mesh: Sequence[int],
    output_file: str | Path | None = None,
    customizer_file: str | Path | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build, save, and close a phonopy band+PDOS Matplotlib Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, metadata = _build_band_with_projected_dos_from_phonopy(
        phonopy_yaml,
        mesh=mesh,
        **kwargs,
    )
    customizer_source = None
    if customizer_file is not None:
        customizer_source = _apply_figure_customizer(figure, customizer_file)
    yaml_source = Path(phonopy_yaml).expanduser().resolve()
    destination = (
        Path(output_file).expanduser().resolve()
        if output_file is not None
        else yaml_source.parent / "phonon_band_projected_dos.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)

    metadata["plot"] = str(destination)
    metadata["customizer"] = (
        str(customizer_source) if customizer_source is not None else None
    )
    metadata_file = destination.with_suffix(".json")
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    metadata["metadata"] = str(metadata_file)
    return metadata


def _apply_figure_customizer(
    figure: "Figure",
    customizer_file: str | Path,
) -> Path:
    import importlib.util

    source = Path(customizer_file).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Plot customizer not found: {source}")
    spec = importlib.util.spec_from_file_location(
        f"_mlp_phonon_plot_customizer_{abs(hash(source))}",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot customizer: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    customize = getattr(module, "customize", None)
    if not callable(customize):
        raise AttributeError(
            f"Plot customizer must define a callable customize(fig): {source}"
        )
    customize(figure)
    return source


def _write_dos_outputs(
    result: dict[str, Any],
    *,
    output_dir: str | Path,
    source_label: str,
    is_gamma_center: bool,
    is_mesh_symmetry: bool,
    dpi: int,
) -> dict[str, Any]:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = result["frequency"]
    dos = result["dos"]
    mesh_numbers = result["mesh"]
    method = result["method"]
    effective_sigma = result["sigma"]
    born_source = result["born"]
    projected_dos = result["projected_dos"]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    png_file = destination / "phonon_dos.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(frequency, dos, linewidth=2, label=source_label)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Frequency (THz)", fontsize=14)
    ax.set_ylabel(r"Phonon DOS (states THz$^{-1}$ primitive cell$^{-1}$)", fontsize=14)
    ax.set_title(
        f"Phonon density of states | mesh "
        f"{mesh_numbers[0]}x{mesh_numbers[1]}x{mesh_numbers[2]}",
        fontsize=16,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(png_file, dpi=dpi)
    plt.close(fig)

    csv_file = destination / "phonon_dos.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frequency_thz", "phonon_dos_states_THz-1_primitive_cell-1"]
        )
        writer.writerows(zip(frequency, dos, strict=True))

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    projected_png_path = destination / "phonon_projected_dos.png"
    projected_csv_path = destination / "phonon_projected_dos.csv"
    projected_png_file = None
    projected_csv_file = None
    projected_integrals = None
    projected_sum_difference = None
    if projected_dos is not None:
        projected_png_file = projected_png_path
        fig, ax = plt.subplots(figsize=(7.5, 5.4), constrained_layout=True)
        ax.plot(
            frequency,
            dos,
            color="black",
            linewidth=1.8,
            linestyle="--",
            label="Total",
        )
        for symbol, element_dos in projected_dos.items():
            ax.plot(frequency, element_dos, linewidth=2, label=symbol)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Frequency (THz)", fontsize=14)
        ax.set_ylabel(
            r"Projected DOS (states THz$^{-1}$ primitive cell$^{-1}$)",
            fontsize=14,
        )
        ax.set_title(
            f"Element-projected phonon DOS | mesh "
            f"{mesh_numbers[0]}x{mesh_numbers[1]}x{mesh_numbers[2]}",
            fontsize=16,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(projected_png_file, dpi=dpi)
        plt.close(fig)

        projected_csv_file = projected_csv_path
        with projected_csv_file.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frequency_thz",
                    "total_dos_states_THz-1_primitive_cell-1",
                    *(
                        f"{symbol}_projected_dos_states_THz-1_primitive_cell-1"
                        for symbol in projected_dos
                    ),
                ]
            )
            writer.writerows(
                zip(frequency, dos, *projected_dos.values(), strict=True)
            )

        projected_integrals = {
            symbol: float(integrate(element_dos, frequency))
            for symbol, element_dos in projected_dos.items()
        }
        projected_sum = np.sum(list(projected_dos.values()), axis=0)
        projected_sum_difference = float(np.max(np.abs(projected_sum - dos)))
    else:
        # Do not leave outputs from an earlier projected run beside metadata
        # that now records projected=false.
        projected_png_path.unlink(missing_ok=True)
        projected_csv_path.unlink(missing_ok=True)

    metadata = {
        "source_label": source_label,
        "input_kind": result["input_kind"],
        "phono3py_yaml": (
            str(result["phono3py_yaml"])
            if result["phono3py_yaml"] is not None
            else None
        ),
        "fc2": str(result["fc2"]) if result["fc2"] is not None else None,
        "phonopy_yaml": (
            str(result["phonopy_yaml"])
            if result["phonopy_yaml"] is not None
            else None
        ),
        "force_constants": (
            str(result["force_constants"])
            if result["force_constants"] is not None
            else None
        ),
        "born": str(born_source) if born_source is not None else None,
        "mesh": mesh_numbers,
        "method": method,
        "sigma_THz": effective_sigma,
        "is_gamma_center": bool(is_gamma_center),
        "is_mesh_symmetry_requested": bool(is_mesh_symmetry),
        "is_mesh_symmetry": bool(result["is_mesh_symmetry"]),
        "frequency_unit": "THz",
        "dos_unit": "states THz^-1 primitive cell^-1",
        "dos_integral": float(integrate(dos, frequency)),
        "projected": projected_dos is not None,
        "projected_by": "element" if projected_dos is not None else None,
        "projected_dos_integrals": projected_integrals,
        "projected_sum_max_abs_difference": projected_sum_difference,
        "projected_plot": (
            str(projected_png_file) if projected_png_file is not None else None
        ),
        "projected_data": (
            str(projected_csv_file) if projected_csv_file is not None else None
        ),
        "plot": str(png_file),
        "data": str(csv_file),
    }
    metadata_file = destination / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    metadata["metadata"] = str(metadata_file)
    return metadata


def plot_dos_comparison(
    dos_results: Sequence[dict[str, Any]],
    *,
    output_dir: str | Path,
    dpi: int = 200,
) -> dict[str, Any]:
    """Overlay independently calculated DOS results without recomputing them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = list(dos_results)
    if len(results) < 2:
        raise ValueError("A DOS comparison requires at least two results.")

    meshes = {tuple(result["mesh"]) for result in results}
    if len(meshes) != 1:
        raise ValueError("All DOS comparison results must use the same mesh.")
    mesh = list(next(iter(meshes)))

    destination = Path(output_dir).expanduser().resolve()
    if any(
        part.lower() == "combined" or part.lower().startswith("combined_")
        for part in destination.parts
    ):
        raise ValueError(f"DOS comparison cannot be written under {destination}.")
    destination.mkdir(parents=True, exist_ok=True)

    curves = []
    for result in results:
        csv_path = Path(result["data"])
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 2:
            raise ValueError(f"DOS data is empty: {csv_path}")
        curves.append(
            {
                "source_label": str(result["source_label"]),
                "source_data": str(csv_path),
                "source_metadata": str(result["metadata"]),
                "fc2": str(result["fc2"]),
                "method": str(result["method"]),
                "sigma_THz": result.get("sigma_THz"),
                "frequency": [float(row[0]) for row in rows[1:]],
                "dos": [float(row[1]) for row in rows[1:]],
            }
        )

    png_file = destination / "phonon_dos.png"
    fig, ax = plt.subplots(figsize=(7.8, 5.5), constrained_layout=True)
    for curve in curves:
        ax.plot(
            curve["frequency"],
            curve["dos"],
            linewidth=2,
            label=curve["source_label"],
        )
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Frequency (THz)", fontsize=14)
    ax.set_ylabel(
        r"Phonon DOS (states THz$^{-1}$ primitive cell$^{-1}$)", fontsize=14
    )
    ax.set_title(f"Phonon density of states | mesh {mesh_tag(mesh)[5:]}", fontsize=16)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(png_file, dpi=dpi)
    plt.close(fig)

    csv_file = destination / "phonon_dos.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_label",
                "frequency_thz",
                "phonon_dos_states_THz-1_primitive_cell-1",
            ]
        )
        for curve in curves:
            writer.writerows(
                (curve["source_label"], frequency, dos)
                for frequency, dos in zip(
                    curve["frequency"], curve["dos"], strict=True
                )
            )

    metadata = {
        "source_labels": [curve["source_label"] for curve in curves],
        "sources": [
            {
                key: curve[key]
                for key in (
                    "source_label",
                    "source_data",
                    "source_metadata",
                    "fc2",
                    "method",
                    "sigma_THz",
                )
            }
            for curve in curves
        ],
        "mesh": mesh,
        "frequency_unit": "THz",
        "dos_unit": "states THz^-1 primitive cell^-1",
        "plot": str(png_file),
        "data": str(csv_file),
    }
    metadata_file = destination / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    metadata["metadata"] = str(metadata_file)
    return metadata


def mesh_tag(mesh: Sequence[int]) -> str:
    values = _validate_mesh(mesh)
    return f"mesh-{values[0]}x{values[1]}x{values[2]}"


def _validate_mesh(mesh: Sequence[int]) -> list[int]:
    values = [int(value) for value in mesh]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("DOS mesh must contain three positive integers.")
    return values
