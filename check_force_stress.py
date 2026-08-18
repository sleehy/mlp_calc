#!/usr/bin/env python3
"""Evaluate the maximum atomic force and residual stress of a POSCAR."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


MLP_CHOICES = ("mattersim", "mace_mp", "sevennet")
VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an MLP single-point calculation for a POSCAR and print the "
            "maximum atomic force and residual stress."
        )
    )
    parser.add_argument("poscar", help="Path to the POSCAR file.")
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Workflow config containing MLP settings (default: config.toml).",
    )
    parser.add_argument(
        "--mlp",
        choices=MLP_CHOICES,
        help="MLP to use (default: workflow.active_mlp in the config).",
    )
    parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def relaunch_in_mlp_env(
    config: dict[str, Any], args: argparse.Namespace
) -> int | None:
    """Relaunch this script in the conda environment configured for the MLP."""
    mlp_name = config["workflow"]["active_mlp"]
    target_env = str(config["mlp"][mlp_name].get("conda_env", "")).strip()
    if (
        args.no_relaunch
        or not target_env
        or os.environ.get("CONDA_DEFAULT_ENV") == target_env
    ):
        return None

    completed = subprocess.run(
        ["conda", "info", "--base"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    conda_sh = Path(completed.stdout.strip()) / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.is_file():
        raise FileNotFoundError(f"conda initialization script not found: {conda_sh}")

    command = [
        "python",
        str(Path(__file__).resolve()),
        str(Path(args.poscar).expanduser().resolve()),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--no-relaunch",
    ]
    if args.mlp:
        command.extend(["--mlp", args.mlp])

    shell_command = " && ".join(
        (
            f"source {shlex.quote(str(conda_sh))}",
            f"conda activate {shlex.quote(target_env)}",
            " ".join(shlex.quote(part) for part in command),
        )
    )
    print(f"[info] Re-launching in conda environment '{target_env}'...", flush=True)
    result = subprocess.run(
        ["bash", "-lc", shell_command],
        cwd=Path(__file__).resolve().parent,
        env=os.environ.copy(),
    )
    return int(result.returncode)


def evaluate(config: dict[str, Any], poscar: Path) -> None:
    import numpy as np
    from ase.io import read
    from ase.units import GPa

    from mlp_phonon_workflow.calculators import build_calculator

    atoms = read(poscar, format="vasp")
    atoms.calc = build_calculator(config)

    # apply_constraint=False reports the actual calculator force even when the
    # POSCAR contains Selective Dynamics constraints.
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    if len(forces) == 0:
        raise ValueError("The POSCAR contains no atoms.")

    force_norms = np.linalg.norm(forces, axis=1)
    max_index = int(np.argmax(force_norms))
    max_force = float(force_norms[max_index])
    max_vector = forces[max_index]

    stress_gpa = np.asarray(atoms.get_stress(voigt=True), dtype=float) / GPa
    pressure_gpa = -float(np.mean(stress_gpa[:3]))
    max_stress_index = int(np.argmax(np.abs(stress_gpa)))

    mlp_name = config["workflow"]["active_mlp"]
    print()
    print(f"POSCAR              : {poscar}")
    print(f"MLP                 : {mlp_name}")
    print(f"Number of atoms     : {len(atoms)}")
    print(f"Potential energy    : {atoms.get_potential_energy():.10f} eV")
    print()
    print(f"Maximum atomic force: {max_force:.10f} eV/Angstrom")
    print(
        "Maximum-force atom  : "
        f"{max_index + 1} ({atoms[max_index].symbol})"
    )
    print(
        "Force vector        : "
        f"[{max_vector[0]: .10f}, {max_vector[1]: .10f}, "
        f"{max_vector[2]: .10f}] eV/Angstrom"
    )
    print()
    print("Residual stress (ASE convention; positive = tension):")
    for label, value in zip(VOIGT_LABELS, stress_gpa, strict=True):
        print(f"  {label}: {value: .10f} GPa")
    print(
        "Maximum |stress|    : "
        f"{abs(stress_gpa[max_stress_index]):.10f} GPa "
        f"({VOIGT_LABELS[max_stress_index]})"
    )
    print(f"Hydrostatic pressure: {pressure_gpa:.10f} GPa (positive = compression)")


def main() -> int:
    args = parse_args()
    poscar = Path(args.poscar).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not poscar.is_file():
        print(f"Error: POSCAR not found: {poscar}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        from mlp_phonon_workflow.config import extra_env, load_config

        config = load_config(config_path, mlp_override=args.mlp)
        for key, value in extra_env(config).items():
            os.environ.setdefault(key, value)

        status = relaunch_in_mlp_env(config, args)
        if status is not None:
            return status

        evaluate(config, poscar)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
