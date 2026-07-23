from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


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
    symprec: float = 1.0e-5,
) -> dict[str, Any]:
    """Calculate harmonic DOS directly from one fc2 input."""
    import numpy as np
    import phono3py
    from phonopy import Phonopy

    mesh_numbers = _validate_mesh(mesh)
    method = str(method).lower()
    if method not in {"tetrahedron", "gaussian"}:
        raise ValueError("DOS method must be 'tetrahedron' or 'gaussian'.")
    if method == "gaussian" and (sigma is None or float(sigma) <= 0):
        raise ValueError("Gaussian DOS requires a positive sigma.")
    effective_sigma = float(sigma) if method == "gaussian" else None
    if frequency_pitch is not None and float(frequency_pitch) <= 0:
        raise ValueError("DOS frequency_pitch must be positive.")

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
    phonon.run_mesh(
        mesh_numbers,
        is_mesh_symmetry=bool(is_mesh_symmetry),
        is_gamma_center=bool(is_gamma_center),
    )
    phonon.run_total_dos(
        sigma=effective_sigma,
        freq_min=frequency_min,
        freq_max=frequency_max,
        freq_pitch=frequency_pitch,
        use_tetrahedron_method=method == "tetrahedron",
    )
    if phonon.total_dos is None:
        raise RuntimeError("Phonopy did not produce a total DOS result.")

    return {
        "frequency": np.asarray(phonon.total_dos.frequency_points, dtype=float),
        "dos": np.asarray(phonon.total_dos.dos, dtype=float),
        "mesh": mesh_numbers,
        "method": method,
        "sigma": effective_sigma,
        "phono3py_yaml": yaml_source,
        "fc2": fc2_source,
        "born": born_source,
    }


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
    symprec: float = 1.0e-5,
    dpi: int = 200,
) -> dict[str, Any]:
    """Calculate and plot harmonic DOS on an independently selected q mesh."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        symprec=symprec,
    )
    frequency = result["frequency"]
    dos = result["dos"]
    mesh_numbers = result["mesh"]
    method = result["method"]
    effective_sigma = result["sigma"]
    yaml_source = result["phono3py_yaml"]
    fc2_source = result["fc2"]
    born_source = result["born"]
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
    metadata = {
        "source_label": source_label,
        "phono3py_yaml": str(yaml_source),
        "fc2": str(fc2_source),
        "born": str(born_source) if born_source is not None else None,
        "mesh": mesh_numbers,
        "method": method,
        "sigma_THz": effective_sigma,
        "is_gamma_center": bool(is_gamma_center),
        "is_mesh_symmetry": bool(is_mesh_symmetry),
        "frequency_unit": "THz",
        "dos_unit": "states THz^-1 primitive cell^-1",
        "dos_integral": float(integrate(dos, frequency)),
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
