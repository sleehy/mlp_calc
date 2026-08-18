from __future__ import annotations

import csv
import json
import os
import warnings
from pathlib import Path
from typing import Any, Sequence

MFP_UNITS = {
    "angstrom": (1.0, r"$\AA$"),
    "nm": (0.1, "nm"),
    "um": (1.0e-4, r"$\mu$m"),
}


def plot_band_dispersion(
    band_yaml_file: str | Path,
    *,
    output_file: str | Path | None = None,
    dpi: int = 200,
) -> dict[str, Any]:
    """Plot a phonopy band.yaml without experimental data."""
    band_data = _read_band_yaml(band_yaml_file)
    destination = _band_plot_destination(
        band_data["source"],
        output_file,
        default_name="phonon_band.png",
    )
    _draw_band_plot(
        band_data,
        destination,
        dpi=dpi,
    )
    return _band_plot_result(
        band_data,
        destination,
        mode="dispersion",
    )


def plot_band_with_experiment(
    band_yaml_file: str | Path,
    experiment_csv_file: str | Path,
    *,
    output_file: str | Path | None = None,
    high_symmetry_positions: Sequence[float] | None = None,
    dpi: int = 200,
) -> dict[str, Any]:
    """Overlay experimental points on a phonopy dispersion relation.

    Experimental x coordinates are assumed to span 0--1 unless explicit
    high-symmetry positions are supplied.
    """
    import numpy as np

    band_data = _read_band_yaml(band_yaml_file)
    experiment_source = Path(experiment_csv_file).expanduser().resolve()
    experiment = _read_experiment_csv(experiment_source)
    positions = (
        np.asarray(high_symmetry_positions, dtype=float)
        if high_symmetry_positions is not None
        else _normalised_band_boundaries(band_data)
    )
    expected = len(band_data["segments"]) + 1
    if positions.shape != (expected,):
        raise ValueError(
            "high_symmetry_positions must contain one value for the first "
            f"point and one for each segment end ({expected} values required)."
        )
    if not np.all(np.isfinite(positions)) or np.any(np.diff(positions) < 0):
        raise ValueError(
            "high_symmetry_positions must be finite and monotonically increasing."
        )

    destination = _band_plot_destination(
        band_data["source"],
        output_file,
        default_name="phonon_band_with_experiment.png",
    )
    _draw_band_plot(
        band_data,
        destination,
        dpi=dpi,
        experiment=experiment,
        plot_boundaries=positions,
    )
    result = _band_plot_result(
        band_data,
        destination,
        mode="experiment",
    )
    result.update(
        {
            "experiment": str(experiment_source),
            "experiment_points": int(len(experiment)),
            "high_symmetry_positions": positions.tolist(),
        }
    )
    return result


def _read_band_yaml(band_yaml_file: str | Path) -> dict[str, Any]:
    import numpy as np
    import yaml

    source = Path(band_yaml_file).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Band YAML file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    phonons = raw.get("phonon")
    if not isinstance(phonons, list) or not phonons:
        raise ValueError(f"No phonon points were found in {source}.")
    try:
        distances = np.asarray([point["distance"] for point in phonons], dtype=float)
        frequencies = np.asarray(
            [
                [band["frequency"] for band in point["band"]]
                for point in phonons
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid phonopy band data in {source}.") from exc
    if frequencies.ndim != 2 or frequencies.shape[0] != len(distances):
        raise ValueError(f"Inconsistent band dimensions in {source}.")
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(frequencies)):
        raise ValueError(f"Non-finite band data found in {source}.")

    segments = _band_segments(raw, distances)
    labels = _band_boundary_labels(raw.get("labels"), len(segments))
    return {
        "source": source,
        "distances": distances,
        "frequencies": frequencies,
        "segments": segments,
        "labels": labels,
    }


def _band_segments(raw: dict[str, Any], distances):
    import numpy as np

    segment_nqpoint = raw.get("segment_nqpoint")
    if isinstance(segment_nqpoint, list):
        counts = [int(value) for value in segment_nqpoint]
        if counts and all(value >= 2 for value in counts) and sum(counts) == len(
            distances
        ):
            segments = []
            start = 0
            for count in counts:
                stop = start + count
                segments.append(np.arange(start, stop, dtype=int))
                start = stop
            return segments

    breaks = np.flatnonzero(np.isclose(np.diff(distances), 0.0, atol=1.0e-12)) + 1
    segments = [
        segment
        for segment in np.split(np.arange(len(distances), dtype=int), breaks)
        if len(segment) >= 2
    ]
    if not segments:
        raise ValueError("Could not determine band-path segments.")
    return segments


def _band_boundary_labels(raw_labels: Any, segment_count: int) -> list[str]:
    if (
        isinstance(raw_labels, list)
        and len(raw_labels) == segment_count
        and all(
            isinstance(pair, (list, tuple)) and len(pair) == 2
            for pair in raw_labels
        )
    ):
        labels = [str(raw_labels[0][0])]
        for index, pair in enumerate(raw_labels):
            right = str(pair[1])
            if index + 1 < segment_count:
                next_left = str(raw_labels[index + 1][0])
                if _plain_band_label(right) != _plain_band_label(next_left):
                    right = f"{right}|{next_left}"
            labels.append(right)
        return labels
    if isinstance(raw_labels, list) and len(raw_labels) == segment_count + 1:
        return [str(label) for label in raw_labels]
    return [""] * (segment_count + 1)


def _plain_band_label(label: str) -> str:
    label = label.strip().strip("$")
    if label.upper() in {"G", "GAMMA", r"\GAMMA"} or label == "Γ":
        return "GAMMA"
    return label


def _display_band_label(label: str) -> str:
    if "|" in label:
        return "|".join(_display_band_label(part) for part in label.split("|"))
    return r"$\Gamma$" if _plain_band_label(label) == "GAMMA" else label


def _normalised_band_boundaries(band_data: dict[str, Any]):
    import numpy as np

    boundaries = _native_band_boundaries(band_data)
    lower = float(boundaries[0])
    span = float(boundaries[-1] - lower)
    if span <= 0:
        return np.linspace(0.0, 1.0, len(boundaries))
    return (boundaries - lower) / span


def _native_band_boundaries(band_data: dict[str, Any]):
    import numpy as np

    distances = band_data["distances"]
    segments = band_data["segments"]
    return np.asarray(
        [distances[segments[0][0]]]
        + [distances[segment[-1]] for segment in segments],
        dtype=float,
    )


def _read_experiment_csv(source: Path):
    import numpy as np

    if not source.exists():
        raise FileNotFoundError(f"Experimental CSV file not found: {source}")
    points = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                point = (float(row[0]), float(row[1]))
            except ValueError:
                continue
            if np.all(np.isfinite(point)):
                points.append(point)
    if not points:
        raise ValueError(f"No numeric x,y experimental points found in {source}.")
    return np.asarray(points, dtype=float)


def _band_plot_destination(
    source: Path,
    output_file: str | Path | None,
    *,
    default_name: str,
) -> Path:
    if output_file is None:
        output = (
            source.parent.parent / "plots" / default_name
            if source.parent.name == "inputs"
            else source.parent / "plots" / default_name
        )
    else:
        output = Path(output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _draw_band_plot(
    band_data: dict[str, Any],
    output: Path,
    *,
    dpi: int,
    experiment=None,
    plot_boundaries=None,
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    native_boundaries = _native_band_boundaries(band_data)
    boundaries = (
        native_boundaries
        if plot_boundaries is None
        else np.asarray(plot_boundaries, dtype=float)
    )
    distances = band_data["distances"]
    frequencies = band_data["frequencies"]

    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for segment_index, segment in enumerate(band_data["segments"]):
        native_left = native_boundaries[segment_index]
        native_right = native_boundaries[segment_index + 1]
        plot_left = boundaries[segment_index]
        plot_right = boundaries[segment_index + 1]
        if np.isclose(native_left, native_right):
            x_values = np.linspace(plot_left, plot_right, len(segment))
        else:
            fraction = (distances[segment] - native_left) / (
                native_right - native_left
            )
            x_values = plot_left + fraction * (plot_right - plot_left)
        for band_index in range(frequencies.shape[1]):
            ax.plot(
                x_values,
                frequencies[segment, band_index],
                linewidth=1,
                color="black",
            )

    if experiment is not None:
        ax.scatter(
            experiment[:, 0],
            experiment[:, 1],
            s=25,
            marker="o",
            label="Experiment",
            zorder=3,
        )
        ax.legend(frameon=False)
    for position in boundaries:
        ax.axvline(position, linewidth=0.6, linestyle="--", color="black")
    ax.axhline(0.0, linewidth=0.7, color="0.35")
    ax.set_xticks(
        boundaries,
        [_display_band_label(label) for label in band_data["labels"]],
    )
    ax.set_xlabel("Wave vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_xlim(float(boundaries[0]), float(boundaries[-1]))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _band_plot_result(
    band_data: dict[str, Any],
    output: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    frequencies = band_data["frequencies"]
    return {
        "source": str(band_data["source"]),
        "plot": str(output),
        "mode": mode,
        "nqpoint": int(len(band_data["distances"])),
        "nband": int(frequencies.shape[1]),
        "minimum_frequency_thz": float(frequencies.min()),
        "imaginary_frequencies_plotted_as_negative": True,
    }


def kappa_plot_kwargs(
    plot_config: dict[str, Any],
    *,
    temperatures: Sequence[float] | None = None,
    bins: int | None = None,
    mfp_unit: str | None = None,
    dpi: int | None = None,
) -> dict[str, Any]:
    """Resolve shared kappa plot settings with optional CLI overrides."""
    return {
        "temperatures": (
            temperatures
            if temperatures is not None
            else plot_config.get("temperatures") or None
        ),
        "bins": int(bins if bins is not None else plot_config.get("bins", 200)),
        "mfp_unit": str(
            mfp_unit if mfp_unit is not None else plot_config.get("mfp_unit", "nm")
        ),
        "dpi": int(dpi if dpi is not None else plot_config.get("dpi", 200)),
    }


def _default_plot_destination(sources: Sequence[Path]) -> Path:
    """Return a plot directory that never uses a combined archive folder."""
    common_parent = Path(os.path.commonpath([str(path.parent) for path in sources]))
    output_root = common_parent.parent if common_parent.name == "inputs" else common_parent
    return output_root / "plots"


def _reject_combined_destination(destination: Path) -> None:
    forbidden = [
        part
        for part in destination.parts
        if part.lower() == "combined" or part.lower().startswith("combined_")
    ]
    if forbidden:
        raise ValueError(
            "Plot output cannot be written inside a combined folder: "
            f"{destination}. Choose a plots directory instead."
        )


def plot_kappa_files(
    hdf5_files: Sequence[str | Path],
    *,
    method: str = "auto",
    output_dir: str | Path | None = None,
    temperatures: Sequence[float] | None = None,
    bins: int = 200,
    mfp_unit: str = "nm",
    dpi: int = 200,
    dos_fc2_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plot conductivity and optionally calculate DOS from a separate fc2."""
    import h5py
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = list(
        dict.fromkeys(Path(path).expanduser().resolve() for path in hdf5_files)
    )
    if not sources:
        raise ValueError("At least one thermal-conductivity HDF5 file is required.")
    missing_sources = [source for source in sources if not source.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Thermal-conductivity HDF5 file not found: "
            + ", ".join(str(source) for source in missing_sources)
        )
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if method not in {"auto", "rta", "lbte"}:
        raise ValueError("method must be 'auto', 'rta', or 'lbte'.")
    if mfp_unit not in MFP_UNITS:
        raise ValueError(
            f"Unknown MFP unit '{mfp_unit}'. Choices: {', '.join(MFP_UNITS)}"
        )

    source_labels = _unique_source_labels(sources)
    source_methods = []
    frequency_upper = 0.0
    for source in sources:
        with h5py.File(source, "r") as h5:
            if "frequency" not in h5:
                raise KeyError(f"Missing HDF5 dataset in {source}: frequency")
            source_frequency = np.asarray(h5["frequency"][:], dtype=float)
            if source_frequency.ndim != 2:
                raise ValueError(
                    "frequency must have shape (grid_point, band), got "
                    + str(source_frequency.shape)
                )
            frequency_upper = max(frequency_upper, float(np.nanmax(source_frequency)))
            source_methods.append(
                _detect_kappa_method(h5, source) if method == "auto" else method
            )
    methods_used = list(dict.fromkeys(source_methods))
    method_tag = methods_used[0] if len(methods_used) == 1 else "mixed"

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
    else:
        destination = _default_plot_destination(sources)
    _reject_combined_destination(destination)
    destination.mkdir(parents=True, exist_ok=True)

    frequency_edges = _linear_edges(np.asarray([frequency_upper]), bins)
    frequency_centres = (frequency_edges[:-1] + frequency_edges[1:]) / 2

    curves = []
    dos_curves = []
    mfp_factor, mfp_unit_label = MFP_UNITS[mfp_unit]

    for source, source_label, source_method in zip(
        sources, source_labels, source_methods, strict=True
    ):
        with h5py.File(source, "r") as h5:
            required = ("frequency", "weight", "mesh", "temperature")
            missing = [name for name in required if name not in h5]
            if missing:
                raise KeyError(
                    f"Missing HDF5 datasets in {source}: {', '.join(missing)}"
                )

            all_temperatures = np.asarray(h5["temperature"][:], dtype=float)
            selected_indices = _temperature_indices(all_temperatures, temperatures)
            selected_temperatures = all_temperatures[selected_indices]

            frequency = np.asarray(h5["frequency"][:], dtype=float)
            if frequency.ndim != 2:
                raise ValueError(
                    "frequency must have shape (grid_point, band), got "
                    + str(frequency.shape)
                )
            if np.any(frequency < -1.0e-8):
                warnings.warn(
                    f"Negative phonon frequencies in {source} were set to zero "
                    "for plotting.",
                    stacklevel=2,
                )
            frequency = np.maximum(frequency, 0.0)

            dos = None
            if dos_fc2_input is None:
                weights = np.asarray(h5["weight"][:], dtype=float)
                dos = _phonon_dos(
                    frequency,
                    weights,
                    frequency_edges,
                    source=source,
                )

            mesh = np.asarray(h5["mesh"][:])
            mesh_count = _mesh_point_count(mesh)
            if source_method == "rta":
                (
                    mode_contribution,
                    total_values,
                    mean_free_path_angstrom,
                    mode_source,
                    mfp_source,
                ) = _read_rta_data(
                    h5,
                    selected_indices,
                    all_temperatures=all_temperatures,
                    frequency_shape=frequency.shape,
                    mesh_count=mesh_count,
                )
            else:
                (
                    mode_contribution,
                    total_values,
                    mean_free_path_angstrom,
                    mode_source,
                    mfp_source,
                ) = _read_lbte_data(
                    h5,
                    selected_indices,
                    all_temperatures=all_temperatures,
                    frequency_shape=frequency.shape,
                    mesh_count=mesh_count,
                )

        curve_method_label = "RTA" if source_method == "rta" else "iterative LBTE"
        compact_label = _compact_legend_label(
            source,
            source_label,
            method=source_method,
            mesh=mesh,
        )
        if dos is not None:
            dos_curves.append(
                {
                    "source": str(source),
                    "source_label": source_label,
                    "method": source_method,
                    "label": compact_label,
                    "frequency_centres": frequency_centres,
                    "dos": dos,
                }
            )

        mean_free_path = mean_free_path_angstrom * mfp_factor
        mfp_sampling = _log_sampling_points(mean_free_path, mode_contribution, bins)

        for row, temperature in enumerate(selected_temperatures):
            contributions = mode_contribution[row]
            histogram = np.histogram(
                frequency.ravel(),
                bins=frequency_edges,
                weights=contributions.ravel(),
            )[0]
            spectral = histogram / np.diff(frequency_edges)
            cumulative_frequency = np.concatenate(([0.0], np.cumsum(histogram)))

            mfp = mean_free_path[row]
            below_range_or_invalid = ~np.isfinite(mfp) | (mfp < mfp_sampling[0])
            baseline = float(contributions[below_range_or_invalid].sum())
            in_range = ~below_range_or_invalid
            mfp_histogram = np.histogram(
                mfp[in_range],
                bins=mfp_sampling,
                weights=contributions[in_range],
            )[0]
            cumulative_mfp = np.concatenate(
                ([baseline], baseline + np.cumsum(mfp_histogram))
            )
            total = None if total_values is None else float(total_values[row])
            curves.append(
                {
                    "source": str(source),
                    "source_label": source_label,
                    "method": source_method,
                    "method_label": curve_method_label,
                    "label": compact_label,
                    "temperature_K": float(temperature),
                    "mesh_point_count": mesh_count,
                    "mode_source": mode_source,
                    "mfp_source": mfp_source,
                    "total_kappa_avg": total,
                    "frequency_centres": frequency_centres,
                    "frequency_edges": frequency_edges,
                    "spectral": spectral,
                    "cumulative_frequency": cumulative_frequency,
                    "mfp_sampling": mfp_sampling,
                    "cumulative_mfp": cumulative_mfp,
                }
            )

    dos_input_kind = "kappa_hdf5"
    if dos_fc2_input is not None:
        from .dos_plot import calculate_dos_from_fc2

        fc2_dos_kwargs = {
            key: value
            for key, value in dos_fc2_input.items()
            if key != "source_label"
        }
        fc2_dos = calculate_dos_from_fc2(**fc2_dos_kwargs)
        fc2_source = Path(fc2_dos["fc2"])
        source_label = str(dos_fc2_input.get("source_label", "RTA fc2"))
        dos_curves = [
            {
                "source": str(fc2_source),
                "source_label": source_label,
                "method": "rta_fc2",
                "label": source_label,
                "frequency_centres": fc2_dos["frequency"],
                "dos": fc2_dos["dos"],
            }
        ]
        dos_input_kind = "rta_fc2"

    method_label = (
        "RTA"
        if methods_used == ["rta"]
        else "iterative LBTE"
        if methods_used == ["lbte"]
        else "RTA + iterative LBTE"
    )
    common_title = f"{method_label} | " + r"$\kappa_{\mathrm{avg}}$"
    dos_png = destination / "phonon_dos.png"
    spectral_png = destination / "spectral_thermal_conductivity.png"
    cumulative_frequency_png = (
        destination / "cumulative_thermal_conductivity_frequency.png"
    )
    cumulative_mfp_png = destination / "cumulative_thermal_conductivity_mfp.png"
    _save_overlay_plot(
        plt,
        [
            (curve["frequency_centres"], curve["dos"], curve["label"])
            for curve in dos_curves
        ],
        xlabel="Frequency (THz)",
        ylabel=r"Phonon DOS (states THz$^{-1}$ unit cell$^{-1}$)",
        title="Phonon density of states",
        output=dos_png,
        dpi=dpi,
    )
    _save_overlay_plot(
        plt,
        [(curve["frequency_centres"], curve["spectral"], curve["label"]) for curve in curves],
        xlabel="Frequency (THz)",
        ylabel=r"Spectral $\kappa_{\mathrm{avg}}$ "
        r"(W m$^{-1}$ K$^{-1}$ THz$^{-1}$)",
        title=f"Spectral thermal conductivity\n{common_title}",
        output=spectral_png,
        dpi=dpi,
    )
    _save_overlay_plot(
        plt,
        [
            (curve["frequency_edges"], curve["cumulative_frequency"], curve["label"])
            for curve in curves
        ],
        xlabel="Frequency (THz)",
        ylabel=r"Cumulative $\kappa_{\mathrm{avg}}$ (W m$^{-1}$ K$^{-1}$)",
        title=f"Cumulative thermal conductivity vs frequency\n{common_title}",
        output=cumulative_frequency_png,
        dpi=dpi,
    )
    _save_overlay_plot(
        plt,
        [(curve["mfp_sampling"], curve["cumulative_mfp"], curve["label"]) for curve in curves],
        xlabel=f"Mean free path ({mfp_unit_label})",
        ylabel=r"Cumulative $\kappa_{\mathrm{avg}}$ (W m$^{-1}$ K$^{-1}$)",
        title=f"Cumulative thermal conductivity vs mean free path\n{common_title}",
        output=cumulative_mfp_png,
        dpi=dpi,
        xscale="log",
    )

    dos_csv = destination / "phonon_dos.csv"
    spectral_csv = destination / "spectral_thermal_conductivity.csv"
    cumulative_frequency_csv = (
        destination / "cumulative_thermal_conductivity_frequency.csv"
    )
    cumulative_mfp_csv = destination / "cumulative_thermal_conductivity_mfp.csv"
    _write_overlay_dos_csv(dos_csv, dos_curves)
    _write_overlay_csv(
        spectral_csv,
        curves,
        x_key="frequency_centres",
        x_name="frequency_thz",
        value_key="spectral",
        value_name="spectral_kappa_avg_W_m-1_K-1_THz-1",
    )
    _write_overlay_csv(
        cumulative_frequency_csv,
        curves,
        x_key="frequency_edges",
        x_name="frequency_thz",
        value_key="cumulative_frequency",
        value_name="cumulative_kappa_avg_W_m-1_K-1",
    )
    _write_overlay_csv(
        cumulative_mfp_csv,
        curves,
        x_key="mfp_sampling",
        x_name=f"mean_free_path_{mfp_unit}",
        value_key="cumulative_mfp",
        value_name="cumulative_kappa_avg_W_m-1_K-1",
    )

    metadata = {
        "sources": [str(source) for source in sources],
        "method": method_tag,
        "methods": methods_used,
        "method_label": method_label,
        "component": "average",
        "frequency_unit": "THz",
        "dos_unit": "states THz^-1 unit cell^-1",
        "dos_normalization": "integral equals the number of phonon bands",
        "dos_input_kind": dos_input_kind,
        "dos_sources": [curve["source"] for curve in dos_curves],
        "mfp_unit": mfp_unit,
        "bins": bins,
        "curves": [
            {
                key: curve[key]
                for key in (
                    "source",
                    "source_label",
                    "method",
                    "method_label",
                    "label",
                    "temperature_K",
                    "mesh_point_count",
                    "mode_source",
                    "mfp_source",
                    "total_kappa_avg",
                )
            }
            for curve in curves
        ],
        "plots": [
            str(dos_png),
            str(spectral_png),
            str(cumulative_frequency_png),
            str(cumulative_mfp_png),
        ],
        "data": [
            str(dos_csv),
            str(spectral_csv),
            str(cumulative_frequency_csv),
            str(cumulative_mfp_csv),
        ],
    }
    metadata_file = destination / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    metadata["metadata"] = str(metadata_file)
    return metadata


def plot_kappa_file(
    hdf5_file: str | Path,
    *,
    method: str,
    output_dir: str | Path | None = None,
    temperatures: Sequence[float] | None = None,
    bins: int = 200,
    mfp_unit: str = "nm",
    dpi: int = 200,
) -> dict[str, Any]:
    """Plot one HDF5 file through the shared multi-file implementation."""
    if method not in {"rta", "lbte"}:
        raise ValueError("method must be 'rta' or 'lbte'.")
    return plot_kappa_files(
        [hdf5_file],
        method=method,
        output_dir=output_dir,
        temperatures=temperatures,
        bins=bins,
        mfp_unit=mfp_unit,
        dpi=dpi,
    )


def _temperature_indices(all_temperatures, requested: Sequence[float] | None):
    import numpy as np

    if not requested:
        return np.arange(len(all_temperatures), dtype=int)
    indices = []
    for target in requested:
        matches = np.flatnonzero(np.isclose(all_temperatures, target, atol=1.0e-6))
        if not len(matches):
            available = ", ".join(f"{value:g}" for value in all_temperatures)
            raise ValueError(
                f"Temperature {target:g} K is not in the file. Available: {available} K"
            )
        index = int(matches[0])
        if index not in indices:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def _mesh_point_count(mesh) -> int:
    import numpy as np

    if mesh.shape == (3,):
        count = int(np.prod(mesh))
    elif mesh.shape == (3, 3):
        count = int(round(abs(np.linalg.det(mesh))))
    else:
        raise ValueError(f"mesh must have shape (3,) or (3, 3), got {mesh.shape}")
    if count <= 0:
        raise ValueError(f"Invalid mesh point count: {count}")
    return count


def _average_voigt(tensor):
    if tensor.shape[-1] != 6:
        raise ValueError(
            f"Conductivity tensor must end in 6 Voigt components: {tensor.shape}"
        )
    return tensor[..., :3].sum(axis=-1) / 3


def _detect_kappa_method(h5, source: Path) -> str:
    if "f_vector" in h5:
        return "lbte"
    if "mode_kappa_RTA" in h5 or "mode_kappa" in h5:
        return "rta"
    raise KeyError(
        f"Could not detect RTA or LBTE data in {source}. Expected 'f_vector', "
        "'mode_kappa_RTA', or 'mode_kappa'."
    )


def _check_mode_sum(
    mode_contribution, total_values, mode_source: str, total_key: str
) -> None:
    import numpy as np

    mode_sum = mode_contribution.sum(axis=(1, 2))
    if not np.allclose(mode_sum, total_values, rtol=1.0e-5, atol=1.0e-8):
        warnings.warn(
            f"Sum of {mode_source} does not match {total_key}: "
            f"mode sum={mode_sum}, total={total_values}",
            stacklevel=2,
        )


def _read_rta_data(
    h5,
    selected_indices,
    *,
    all_temperatures,
    frequency_shape: tuple[int, int],
    mesh_count: int,
):
    import numpy as np

    if "mode_kappa_RTA" in h5:
        mode_key = "mode_kappa_RTA"
        total_key = "kappa_RTA"
    elif "mode_kappa" in h5:
        mode_key = "mode_kappa"
        total_key = "kappa"
    else:
        raise KeyError(
            "RTA plotting requires 'mode_kappa' or 'mode_kappa_RTA' in the "
            "kappa HDF5 file."
        )

    mode_tensor = np.asarray(h5[mode_key][:], dtype=float)
    if mode_tensor.ndim == 3:
        mode_tensor = mode_tensor[np.newaxis, ...]
    expected_shape = (len(all_temperatures), *frequency_shape, 6)
    if mode_tensor.shape != expected_shape:
        raise ValueError(
            f"{mode_key} must have shape {expected_shape}, got {mode_tensor.shape}"
        )
    mode_contribution = _average_voigt(mode_tensor)[selected_indices] / mesh_count

    total_values = None
    if total_key in h5:
        total_tensor = np.asarray(h5[total_key][:], dtype=float)
        if total_tensor.ndim == 1:
            total_tensor = total_tensor[np.newaxis, ...]
        total_values = _average_voigt(total_tensor)[selected_indices]
        _check_mode_sum(mode_contribution, total_values, mode_key, total_key)

    mfp, mfp_source = _read_rta_mean_free_path(
        h5, selected_indices, frequency_shape=frequency_shape
    )
    return mode_contribution, total_values, mfp, mode_key, mfp_source


def _read_rta_mean_free_path(h5, selected_indices, *, frequency_shape):
    import numpy as np

    for name in ("gamma", "group_velocity"):
        if name not in h5:
            raise KeyError(
                f"'{name}' is required to derive mean free path when a compatible "
                "mean_free_path dataset is unavailable."
            )
    gamma = np.asarray(h5["gamma"][:], dtype=float)
    if gamma.ndim == 2:
        gamma = gamma[np.newaxis, ...]
    group_velocity = np.asarray(h5["group_velocity"][:], dtype=float)
    if group_velocity.shape != (*frequency_shape, 3):
        raise ValueError(
            "group_velocity must have shape (grid_point, band, 3), got "
            + str(group_velocity.shape)
        )
    if gamma.shape[1:] != frequency_shape:
        raise ValueError(
            "gamma must have shape (temperature, grid_point, band), got "
            + str(gamma.shape)
        )

    effective_gamma = gamma.copy()
    if "gamma_isotope" in h5:
        gamma_isotope = np.asarray(h5["gamma_isotope"][:], dtype=float)
        if gamma_isotope.ndim == 2:
            effective_gamma += gamma_isotope[np.newaxis, ...]
        elif gamma_isotope.shape == gamma.shape:
            effective_gamma += gamma_isotope
        else:
            warnings.warn(
                f"Ignoring gamma_isotope with unexpected shape {gamma_isotope.shape}.",
                stacklevel=2,
            )

    velocity_norm = np.linalg.norm(group_velocity, axis=-1)
    effective_gamma = effective_gamma[selected_indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        mfp = np.where(
            effective_gamma > 0,
            velocity_norm[np.newaxis, ...] / (4 * np.pi * effective_gamma),
            0.0,
        )
    return mfp, "|group_velocity| / (4*pi*effective_gamma)"


def _read_lbte_data(
    h5,
    selected_indices,
    *,
    all_temperatures,
    frequency_shape: tuple[int, int],
    mesh_count: int,
):
    import numpy as np
    from phonopy.physical_units import get_physical_units

    required = (
        "f_vector",
        "group_velocity",
        "heat_capacity",
        "weight",
        "kappa_unit_conversion",
        "kappa",
    )
    missing = [name for name in required if name not in h5]
    if missing:
        raise KeyError(
            "LBTE plotting requires these HDF5 datasets: " + ", ".join(missing)
        )

    f_vector = np.asarray(h5["f_vector"][:], dtype=float)
    if f_vector.ndim == 3:
        if len(all_temperatures) != 1:
            raise ValueError(
                "This LBTE file contains multiple temperatures but f_vector has no "
                "temperature dimension. Run and save LBTE separately for each "
                "temperature, or save temperature-resolved f_vector data."
            )
        f_vector = f_vector[np.newaxis, ...]
    expected_f_shape = (len(all_temperatures), *frequency_shape, 3)
    if f_vector.shape != expected_f_shape:
        raise ValueError(
            f"f_vector must have shape {expected_f_shape}, got {f_vector.shape}"
        )
    f_vector = f_vector[selected_indices]

    group_velocity = np.asarray(h5["group_velocity"][:], dtype=float)
    if group_velocity.shape != (*frequency_shape, 3):
        raise ValueError(
            "group_velocity must have shape (grid_point, band, 3), got "
            + str(group_velocity.shape)
        )

    heat_capacity = np.asarray(h5["heat_capacity"][:], dtype=float)
    if heat_capacity.ndim == 2:
        heat_capacity = heat_capacity[np.newaxis, ...]
    expected_cv_shape = (len(all_temperatures), *frequency_shape)
    if heat_capacity.shape != expected_cv_shape:
        raise ValueError(
            f"heat_capacity must have shape {expected_cv_shape}, got "
            + str(heat_capacity.shape)
        )
    heat_capacity = heat_capacity[selected_indices]

    weights = np.asarray(h5["weight"][:], dtype=float)
    if weights.shape != (frequency_shape[0],):
        raise ValueError(
            f"weight must have shape ({frequency_shape[0]},), got {weights.shape}"
        )

    selected_temperatures = np.asarray(all_temperatures[selected_indices], dtype=float)
    kb = float(get_physical_units().KB)
    valid = heat_capacity > 1.0e-10
    scale = np.zeros_like(heat_capacity)
    # This is the same LBTE MFP definition used by phono3py's
    # CollisionMatrixKernel._set_mean_free_path. f_vector follows phono3py's
    # sign convention, so the conductivity expression below has a minus sign.
    with np.errstate(divide="ignore", invalid="ignore"):
        scale[valid] = (
            -np.broadcast_to(
                selected_temperatures[:, np.newaxis, np.newaxis],
                heat_capacity.shape,
            )[valid]
            * np.sqrt(kb / heat_capacity[valid])
            / np.pi
        )
    mean_free_path_vector = scale[..., np.newaxis] * f_vector
    mean_free_path = np.linalg.norm(mean_free_path_vector, axis=-1)

    conversion = float(np.asarray(h5["kappa_unit_conversion"][()]))
    velocity_dot_mfp = np.sum(
        group_velocity[np.newaxis, ...] * mean_free_path_vector, axis=-1
    )
    # The trace is invariant under the k-star rotations. Therefore kappa_avg
    # can be reconstructed directly with the irreducible-grid multiplicities;
    # individual xx, yy, zz components would additionally need the rotations.
    mode_contribution = (
        -2
        * np.pi
        * conversion
        * heat_capacity
        * velocity_dot_mfp
        / 3
        * weights[np.newaxis, :, np.newaxis]
        / mesh_count
    )

    total_tensor = np.asarray(h5["kappa"][:], dtype=float)
    if total_tensor.ndim == 1:
        total_tensor = total_tensor[np.newaxis, ...]
    total_values = _average_voigt(total_tensor)[selected_indices]
    mode_source = "LBTE kappa_avg reconstructed from f_vector"
    _check_mode_sum(mode_contribution, total_values, mode_source, "kappa")
    return (
        mode_contribution,
        total_values,
        mean_free_path,
        mode_source,
        "LBTE mean free path reconstructed from f_vector",
    )


def _linear_edges(values, bins: int):
    import numpy as np

    upper = float(np.nanmax(values))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("No positive finite phonon frequency was found.")
    return np.linspace(0.0, upper * (1.0 + 1.0e-12), bins + 1)


def _compact_legend_label(
    source: Path,
    fallback_label: str,
    *,
    method: str,
    mesh,
) -> str:
    """Return ``mlp/method/NxNxN`` for plot legends."""
    parts = source.parts
    mlp_name = None
    if "thermal_conductivity" in parts:
        index = parts.index("thermal_conductivity")
        if index > 0:
            mlp_name = parts[index - 1]
    if mlp_name is None and "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            mlp_name = parts[index + 1]
    if mlp_name is None:
        mlp_name = fallback_label.split("/", maxsplit=1)[0]

    mesh_label = "x".join(str(int(value)) for value in mesh.reshape(-1))
    return f"{mlp_name}/{method}/{mesh_label}"


def _phonon_dos(frequency, weights, frequency_edges, *, source: Path):
    """Return a weighted histogram whose integral is the number of bands."""
    import numpy as np

    expected_shape = (frequency.shape[0],)
    if weights.shape != expected_shape:
        raise ValueError(
            f"weight in {source} must have shape {expected_shape}, got "
            f"{weights.shape}"
        )
    if not np.all(np.isfinite(frequency)):
        raise ValueError(f"Non-finite phonon frequency found in {source}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"Phonon q-point weights in {source} must be finite and >= 0.")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError(f"Phonon q-point weights in {source} sum to zero.")

    mode_weights = np.broadcast_to(weights[:, np.newaxis], frequency.shape)
    histogram = np.histogram(
        frequency.ravel(),
        bins=frequency_edges,
        weights=mode_weights.ravel(),
    )[0]
    return histogram / weight_sum / np.diff(frequency_edges)


def _log_sampling_points(values, contributions, bins: int):
    import numpy as np

    finite_positive = np.isfinite(values) & (values > 0)
    max_contribution = float(np.nanmax(np.abs(contributions)))
    significant = np.abs(contributions) > max(max_contribution * 1.0e-12, 1.0e-30)
    relevant = values[finite_positive & significant]
    all_positive = values[finite_positive]
    if not all_positive.size:
        raise ValueError("No positive finite mean free path was found.")
    lower = float(relevant.min()) if relevant.size else float(all_positive.min())
    upper = float(all_positive.max())
    if np.isclose(lower, upper):
        lower /= 10
        upper *= 10
    return np.geomspace(lower, upper * (1.0 + 1.0e-12), bins + 1)


def _save_overlay_plot(
    plt,
    curves,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    dpi: int,
    xscale: str = "linear",
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8), constrained_layout=True)
    for x_values, y_values, label in curves:
        ax.plot(x_values, y_values, linewidth=2, label=label)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xscale(xscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _unique_source_labels(sources: Sequence[Path]) -> list[str]:
    path_parts = [source.with_suffix("").parts for source in sources]
    depth = [1] * len(sources)
    labels = [parts[-1] for parts in path_parts]
    while len(set(labels)) != len(labels):
        duplicate_labels = {
            label for label in labels if labels.count(label) > 1
        }
        changed = False
        for index, label in enumerate(labels):
            if label in duplicate_labels and depth[index] < len(path_parts[index]):
                depth[index] += 1
                labels[index] = "/".join(path_parts[index][-depth[index] :])
                changed = True
        if not changed:
            break
    return labels


def _write_overlay_csv(
    output: Path,
    curves,
    *,
    x_key: str,
    x_name: str,
    value_key: str,
    value_name: str,
) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "source_label", "method", "temperature_K", x_name, value_name]
        )
        for curve in curves:
            for x_value, value in zip(curve[x_key], curve[value_key], strict=True):
                writer.writerow(
                    [
                        curve["source"],
                        curve["source_label"],
                        curve["method"],
                        curve["temperature_K"],
                        x_value,
                        value,
                    ]
                )


def _write_overlay_dos_csv(output: Path, curves) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "source_label",
                "method",
                "frequency_thz",
                "phonon_dos_states_THz-1_unit_cell-1",
            ]
        )
        for curve in curves:
            for frequency, dos in zip(
                curve["frequency_centres"], curve["dos"], strict=True
            ):
                writer.writerow(
                    [
                        curve["source"],
                        curve["source_label"],
                        curve["method"],
                        frequency,
                        dos,
                    ]
                )
