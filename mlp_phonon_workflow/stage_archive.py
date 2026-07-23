from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import active_mlp, plot_archive_root
from .stage_common import _copy_existing_files, stage_paths


def _plot_kappa_outputs(
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    method: str,
) -> list[str]:
    output_path = Path(output_dir)
    plot_outputs = []
    thermal_config = config["thermal_conductivity"]
    plot_config = thermal_config.get("plots", {})
    if not bool(plot_config.get("enabled", True)):
        return plot_outputs

    from .kappa_plot import kappa_plot_kwargs, plot_kappa_files

    hdf5_paths = sorted(output_path.glob("kappa-*.hdf5"))
    if not hdf5_paths:
        return plot_outputs
    plot_result = plot_kappa_files(
        hdf5_paths,
        method=method,
        output_dir=output_path / "plots",
        dos_fc2_input=_rta_fc2_dos_input(config),
        **kappa_plot_kwargs(plot_config),
    )
    plot_outputs.extend(plot_result["plots"])
    return plot_outputs


def _archive_band_outputs(
    config: dict[str, Any], output_dir: str | Path
) -> list[str]:
    archive_config = config.get("plot_archive", {})
    if not bool(archive_config.get("enabled", True)):
        return []

    source_dir = Path(output_dir)
    destination = (
        plot_archive_root(config) / active_mlp(config) / "band" / "inputs"
    )
    overwrite = bool(archive_config.get("overwrite", True))
    archived_paths = _copy_existing_files(
        (
            source_dir / "band.yaml",
            source_dir / "phonopy_params.yaml",
            source_dir / "force_constants.hdf5",
            source_dir / "FORCE_CONSTANTS",
        ),
        destination,
        overwrite=overwrite,
    )
    archived = [str(path) for path in archived_paths]

    if archived:
        print(f"[plot-archive] band inputs -> {destination}")
    return archived


def _archive_dos_inputs(config: dict[str, Any]) -> list[str]:
    archive_config = config.get("plot_archive", {})
    if not bool(archive_config.get("enabled", True)):
        return []

    fc_paths = stage_paths(config, "ph3-fc")
    destination = plot_archive_root(config) / active_mlp(config) / "dos" / "inputs"
    overwrite = bool(archive_config.get("overwrite", True))
    sources = (
        fc_paths.output / "phono3py_params.yaml",
        fc_paths.output / "fc2.hdf5",
        fc_paths.input / "BORN",
    )
    archived_paths = _copy_existing_files(
        sources, destination, overwrite=overwrite
    )
    archived = [str(path) for path in archived_paths]

    required_names = {"phono3py_params.yaml", "fc2.hdf5"}
    archived_names = {Path(path).name for path in archived}
    if archived and not required_names.issubset(archived_names):
        missing = ", ".join(sorted(required_names - archived_names))
        raise FileNotFoundError(f"Missing DOS archive input(s): {missing}")
    if archived:
        print(f"[plot-archive] DOS inputs -> {destination}")
    return archived


def plot_dos_archive(
    config: dict[str, Any],
    *,
    mesh: Iterable[int] | None = None,
    method: str | None = None,
    sigma: float | None = None,
    frequency_min: float | None = None,
    frequency_max: float | None = None,
    frequency_pitch: float | None = None,
    dpi: int | None = None,
) -> dict[str, Any]:
    from .dos_plot import mesh_tag, plot_dos_from_fc2

    if not bool(config.get("plot_archive", {}).get("enabled", True)):
        raise ValueError("plot_archive must be enabled to run plot-dos.")

    overrides = {
        "mesh": mesh,
        "method": method,
        "sigma": sigma,
        "frequency_min": frequency_min,
        "frequency_max": frequency_max,
        "frequency_pitch": frequency_pitch,
        "dpi": dpi,
    }
    dos_settings = {
        **config.get("dos", {}),
        **{key: value for key, value in overrides.items() if value is not None},
    }
    mesh_numbers = [int(value) for value in dos_settings.get("mesh", [40, 40, 40])]

    archive_root = plot_archive_root(config)
    mlp_name = active_mlp(config)
    source_dir = archive_root / mlp_name / "dos" / "inputs"
    yaml_source = source_dir / "phono3py_params.yaml"
    fc2_source = source_dir / "fc2.hdf5"
    if not yaml_source.exists() or not fc2_source.exists():
        _archive_dos_inputs(config)
    for source in (yaml_source, fc2_source):
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run ph3-fc first to prepare DOS inputs."
            )

    destination = archive_root / mlp_name / "dos" / mesh_tag(mesh_numbers)
    input_dir = destination / "inputs"
    overwrite = bool(config.get("plot_archive", {}).get("overwrite", True))
    _copy_existing_files(
        (yaml_source, fc2_source, source_dir / "BORN"),
        input_dir,
        overwrite=overwrite,
    )

    ph3_config = config["phono3py"]
    result = plot_dos_from_fc2(
        input_dir / "phono3py_params.yaml",
        input_dir / "fc2.hdf5",
        mesh=mesh_numbers,
        output_dir=destination,
        source_label=mlp_name,
        calculator=str(ph3_config.get("calculator", "vasp")),
        born_file=(input_dir / "BORN") if (input_dir / "BORN").exists() else None,
        method=str(dos_settings.get("method", "tetrahedron")),
        sigma=dos_settings.get("sigma"),
        frequency_min=dos_settings.get("frequency_min"),
        frequency_max=dos_settings.get("frequency_max"),
        frequency_pitch=dos_settings.get("frequency_pitch"),
        is_gamma_center=bool(dos_settings.get("is_gamma_center", True)),
        is_mesh_symmetry=bool(dos_settings.get("is_mesh_symmetry", True)),
        symprec=float(ph3_config.get("symprec", 1.0e-5)),
        dpi=int(dos_settings.get("dpi", 200)),
    )

    print(f"[plot-dos] {mlp_name} mesh {mesh_numbers} -> {destination}")
    return result


def _archive_kappa_outputs(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    method: str,
) -> list[str]:
    archive_config = config.get("plot_archive", {})
    if not bool(archive_config.get("enabled", True)):
        return []

    source_dir = Path(output_dir)
    thermal_destination = (
        plot_archive_root(config)
        / active_mlp(config)
        / "thermal_conductivity"
    )
    destination = thermal_destination / method
    input_dir = destination / "inputs"
    overwrite = bool(archive_config.get("overwrite", True))
    archived_hdf5 = _copy_existing_files(
        sorted(source_dir.glob("kappa-*.hdf5")),
        input_dir,
        overwrite=overwrite,
    )

    if not archived_hdf5:
        return []

    archived = [str(path) for path in archived_hdf5]
    plot_config = config["thermal_conductivity"].get("plots", {})
    if bool(plot_config.get("enabled", True)):
        from .kappa_plot import kappa_plot_kwargs, plot_kappa_files

        all_archived_hdf5 = sorted(
            thermal_destination.glob("*/inputs/kappa-*.hdf5")
        )
        plot_result = plot_kappa_files(
            all_archived_hdf5,
            method="auto",
            output_dir=thermal_destination / "plots",
            dos_fc2_input=_rta_fc2_dos_input(config),
            **kappa_plot_kwargs(plot_config),
        )
        archived.extend(plot_result["plots"])
        archived.extend(plot_result["data"])
        archived.append(plot_result["metadata"])

    print(
        f"[plot-archive] {method} inputs -> {input_dir}; "
        f"all-method plots -> {thermal_destination / 'plots'}"
    )
    return archived


def _rta_fc2_dos_input(config: dict[str, Any]) -> dict[str, Any]:
    """Return the single RTA-side fc2 input used by conductivity DOS plots."""
    rta_input = stage_paths(config, "kappa-rta").input
    yaml_source = rta_input / "phono3py_params.yaml"
    if not yaml_source.exists():
        yaml_source = rta_input / "phono3py_disp.yaml"
    fc2_source = rta_input / "fc2.hdf5"
    for source in (yaml_source, fc2_source):
        if not source.exists():
            raise FileNotFoundError(
                f"Missing RTA DOS input: {source}. Populate kappa-rta inputs first."
            )

    dos_config = config.get("dos", {})
    ph3_config = config["phono3py"]
    mesh = config["thermal_conductivity"].get("mesh", [8, 8, 8])
    mesh_label = "x".join(str(int(value)) for value in mesh)
    return {
        "phono3py_yaml": yaml_source,
        "fc2_file": fc2_source,
        "mesh": mesh,
        "calculator": str(ph3_config.get("calculator", "vasp")),
        "born_file": (rta_input / "BORN") if (rta_input / "BORN").exists() else None,
        "method": str(dos_config.get("method", "tetrahedron")),
        "sigma": dos_config.get("sigma"),
        "frequency_min": dos_config.get("frequency_min"),
        "frequency_max": dos_config.get("frequency_max"),
        "frequency_pitch": dos_config.get("frequency_pitch"),
        "is_gamma_center": bool(dos_config.get("is_gamma_center", True)),
        "is_mesh_symmetry": bool(dos_config.get("is_mesh_symmetry", True)),
        "symprec": float(ph3_config.get("symprec", 1.0e-5)),
        "source_label": f"{active_mlp(config)}/rta/{mesh_label}",
    }
