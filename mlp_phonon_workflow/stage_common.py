from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import run_dir


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
    _populate_relaxed_input(config, "displace")


def _populate_ph3_displace_input(config: dict[str, Any]) -> None:
    _populate_relaxed_input(config, "ph3-displace")


def _populate_relaxed_input(config: dict[str, Any], stage: str) -> None:
    relax_out = stage_paths(config, "relax").output / "POSCAR"
    _copy_existing_files(
        (relax_out,),
        stage_paths(config, stage).input,
        overwrite=_overwrite_next_inputs(config),
    )


def _populate_forces_input(config: dict[str, Any]) -> None:
    displace_paths = stage_paths(config, "displace")
    forces_in = stage_paths(config, "forces").input
    overwrite = _overwrite_next_inputs(config)
    sources = [
        *(
            displace_paths.output / name
            for name in ("POSCAR", "SPOSCAR", "phonopy_disp.yaml")
        ),
        *_displacement_poscars(displace_paths.output),
    ]
    _copy_existing_files(sources, forces_in, overwrite=overwrite)


def _populate_band_input(config: dict[str, Any]) -> None:
    forces_out = stage_paths(config, "forces").output
    forces_in = stage_paths(config, "forces").input
    displace_in = stage_paths(config, "displace").input
    band_in = stage_paths(config, "band").input
    overwrite = _overwrite_next_inputs(config)

    _copy_existing_files(
        (
            forces_out / "FORCE_SETS",
            forces_in / "phonopy_disp.yaml",
            displace_in / "POSCAR",
            forces_out / "phonopy_params.yaml",
        ),
        band_in,
        overwrite=overwrite,
    )


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
    sources = [
        *(source_paths.output / name for name in fixed_names),
        *_ph3_displacement_poscars(source_paths.output),
        source_paths.input / "BORN",
    ]
    _copy_existing_files(sources, destination, overwrite=overwrite)


def _populate_ph3_fc_input(config: dict[str, Any]) -> None:
    force_paths = stage_paths(config, "ph3-forces")
    destination = stage_paths(config, "ph3-fc").input
    overwrite = _overwrite_next_inputs(config)
    _copy_existing_files(
        (
            force_paths.input / "phono3py_disp.yaml",
            force_paths.output / "forces_fc3.npy",
            force_paths.output / "forces_fc2.npy",
            force_paths.output / "energies_fc3.npy",
            force_paths.output / "energies_fc2.npy",
            force_paths.input / "BORN",
        ),
        destination,
        overwrite=overwrite,
    )


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
        _copy_existing_files(
            sources, stage_paths(config, stage).input, overwrite=overwrite
        )


def _overwrite_next_inputs(config: dict[str, Any]) -> bool:
    return bool(config["workflow"].get("overwrite_next_inputs", False))


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        src_stat = src.stat()
        dst_stat = dst.stat()
        if (
            src_stat.st_size == dst_stat.st_size
            and src_stat.st_mtime_ns == dst_stat.st_mtime_ns
        ):
            return
    shutil.copy2(src, dst)


def _copy_existing_files(
    sources: Iterable[Path], destination: Path, *, overwrite: bool
) -> list[Path]:
    copied = []
    for source in sources:
        if not source.exists():
            continue
        target = destination / source.name
        _copy_file(source, target, overwrite=overwrite)
        copied.append(target)
    return copied


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
