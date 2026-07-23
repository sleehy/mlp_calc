from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .stage_common import (
    StagePaths,
    _ph3_displacement_poscars,
    _safe_float,
)


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
