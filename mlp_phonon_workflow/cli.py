from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import (
    ConfigError,
    active_mlp,
    conda_env_for_stage,
    extra_env,
    load_config,
    plot_archive_root,
    run_dir,
)
from .stages import PHONOPY_STAGES, STAGES, run_stage, stage_status

MLP_CHOICES = ("mattersim", "mace_mp", "sevennet")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "plot-kappa":
            config = load_config(args.config)
            _apply_extra_env(config)
            return _plot_kappa(args, config)

        if args.command == "plot-dos":
            mlp_override = args.mlp[0] if args.mlp else None
            config = load_config(args.config, mlp_override=mlp_override)
            _apply_extra_env(config)
            status = _maybe_relaunch_plot_dos(config, args)
            if status is not None:
                return status
            return _plot_dos(args, config)

        config = load_config(
            args.config, mlp_override=args.mlp, poscar_override=args.poscar
        )
        _apply_extra_env(config)
        if args.command == "status":
            _print_status(config)
            return 0

        if args.stage == "all":
            stages = PHONOPY_STAGES
        elif args.stage == "phono3py-all":
            stages = _phono3py_stages(config)
        else:
            stages = (args.stage,)
        for stage in stages:
            status = _maybe_relaunch(config, stage, args)
            if status is not None:
                if status != 0:
                    return status
                continue
            print(f"[workflow] run {stage} in {run_dir(config)}")
            run_stage(config, stage, force=args.force)
        return 0
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlp-phonon-workflow",
        description=(
            "Resumable MLP workflow for phonopy bands and phono3py thermal "
            "conductivity."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one stage or all stages.")
    run_parser.add_argument(
        "stage",
        choices=[*STAGES, "all", "phono3py-all"],
        help=(
            "Stage to run. 'all' keeps the original phonopy chain; "
            "'phono3py-all' runs relaxation through configured conductivity methods."
        ),
    )
    _add_common_args(run_parser)
    run_parser.add_argument(
        "--force", action="store_true", help="Rerun even if output exists."
    )
    run_parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not auto-relaunch into the configured conda env.",
    )

    status_parser = subparsers.add_parser("status", help="Show resumable stage status.")
    _add_common_args(status_parser)

    dos_parser = subparsers.add_parser(
        "plot-dos",
        help="Calculate harmonic phonon DOS on an independently selected q mesh.",
    )
    dos_parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="DOS q-point mesh. Default: dos.mesh from the config.",
    )
    dos_parser.add_argument(
        "--method",
        choices=["tetrahedron", "gaussian"],
        help="DOS integration method. Default: dos.method from the config.",
    )
    dos_parser.add_argument(
        "--sigma",
        type=float,
        help="Gaussian width in THz; required for method=gaussian.",
    )
    dos_parser.add_argument(
        "--frequency-min", type=float, help="Minimum frequency in THz."
    )
    dos_parser.add_argument(
        "--frequency-max", type=float, help="Maximum frequency in THz."
    )
    dos_parser.add_argument(
        "--frequency-pitch", type=float, help="Frequency sampling pitch in THz."
    )
    dos_parser.add_argument("--dpi", type=int, help="Plot DPI. Default: dos.dpi.")
    _add_config_arg(dos_parser)
    dos_parser.add_argument(
        "--mlp",
        nargs="+",
        choices=MLP_CHOICES,
        help=(
            "One or more MLPs whose archived fc2 files are used. Multiple MLPs "
            "are overlaid in one comparison plot. Default: workflow.active_mlp."
        ),
    )
    dos_parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Multi-MLP comparison output root. Default: "
            "<plot_archive>/plots/dos/."
        ),
    )
    dos_parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not relaunch into the selected MLP conda environment.",
    )

    plot_parser = subparsers.add_parser(
        "plot-kappa",
        help="Plot phonon DOS and conductivity from kappa HDF5 files.",
    )
    plot_parser.add_argument("hdf5", nargs="+", help="phono3py kappa-*.hdf5 file(s).")
    plot_parser.add_argument(
        "--method",
        choices=["auto", "rta", "lbte"],
        default="auto",
        help=(
            "Data method. 'auto' detects each file independently and permits RTA "
            "and LBTE curves in the same plots (default)."
        ),
    )
    plot_parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Output root. The overlaid figures are written under "
            "<root>/plots/ without method-specific subdirectories."
        ),
    )
    plot_parser.add_argument(
        "-T",
        "--temperature",
        action="append",
        type=float,
        help=(
            "Temperature in K to plot. Repeat for multiple values; default comes "
            "from thermal_conductivity.plots.temperatures."
        ),
    )
    _add_config_arg(plot_parser)
    plot_parser.add_argument(
        "--bins",
        type=int,
        help=(
            "Number of plot bins. Overrides thermal_conductivity.plots.bins "
            "from the config."
        ),
    )
    plot_parser.add_argument(
        "--mfp-unit",
        choices=["angstrom", "nm", "um"],
        help=(
            "Mean-free-path unit. Overrides "
            "thermal_conductivity.plots.mfp_unit from the config."
        ),
    )
    plot_parser.add_argument(
        "--dpi",
        type=int,
        help="Plot DPI. Overrides thermal_conductivity.plots.dpi from the config.",
    )
    return parser


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default="config.toml",
        help="Path to TOML, YAML, or JSON config. Default: config.toml",
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    _add_config_arg(parser)
    parser.add_argument(
        "--mlp",
        choices=MLP_CHOICES,
        help="Override workflow.active_mlp from config.",
    )
    parser.add_argument(
        "--poscar",
        help="Override workflow.input_poscar for the relax stage.",
    )


def _maybe_relaunch(config: dict, stage: str, args: argparse.Namespace) -> int | None:
    if args.no_relaunch or not bool(config["execution"].get("auto_relaunch", True)):
        return None

    target_env = conda_env_for_stage(config, stage)
    if not target_env:
        return None
    if os.environ.get("CONDA_DEFAULT_ENV") == target_env:
        return None

    conda_base = _conda_base()
    conda_sh = Path(conda_base) / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise ConfigError(f"conda.sh not found: {conda_sh}")

    command = _relaunch_command(conda_sh, target_env, stage, args)
    env = os.environ.copy()
    env.update(extra_env(config))
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"[workflow] relaunch {stage} in conda env '{target_env}'", flush=True)
    completed = subprocess.run(["bash", "-lc", command], cwd=repo_root, env=env)
    return int(completed.returncode)


def _relaunch_command(
    conda_sh: Path,
    target_env: str,
    stage: str,
    args: argparse.Namespace,
) -> str:
    cmd = [
        "python",
        "-m",
        "mlp_phonon_workflow",
        "run",
        stage,
        "--config",
        str(Path(args.config).resolve()),
        "--no-relaunch",
    ]
    if args.force:
        cmd.append("--force")
    if args.mlp:
        cmd.extend(["--mlp", args.mlp])
    if args.poscar:
        cmd.extend(["--poscar", str(Path(args.poscar).resolve())])

    return " && ".join(
        [
            f"source {shlex.quote(str(conda_sh))}",
            f"conda activate {shlex.quote(target_env)}",
            " ".join(shlex.quote(part) for part in cmd),
        ]
    )


def _conda_base() -> str:
    completed = subprocess.run(
        ["conda", "info", "--base"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _apply_extra_env(config: dict) -> None:
    for key, value in extra_env(config).items():
        os.environ.setdefault(key, value)


def _print_status(config: dict) -> None:
    print(f"run_dir: {run_dir(config)}")
    for row in stage_status(config):
        print(f"{row['stage']:16s} {row['status']:8s} {row['output']}")


def _phono3py_stages(config: dict) -> tuple[str, ...]:
    stages = ["relax", "ph3-displace", "ph3-forces", "ph3-fc"]
    methods = config["thermal_conductivity"].get("methods", ["rta", "iterative"])
    if "rta" in methods:
        stages.append("kappa-rta")
    if "iterative" in methods:
        stages.append("kappa-iterative")
    return tuple(stages)


def _plot_kappa(args: argparse.Namespace, config: dict) -> int:
    from .kappa_plot import kappa_plot_kwargs, plot_kappa_files

    plot_config = config["thermal_conductivity"].get("plots", {})
    inputs = [Path(filename).expanduser().resolve() for filename in args.hdf5]
    output_root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    )
    destination = None if output_root is None else output_root / "plots"
    result = plot_kappa_files(
        inputs,
        method=args.method,
        output_dir=destination,
        **kappa_plot_kwargs(
            plot_config,
            temperatures=args.temperature,
            bins=args.bins,
            mfp_unit=args.mfp_unit,
            dpi=args.dpi,
        ),
    )
    print("[plot-kappa] inputs:")
    for source in inputs:
        print(f"  {source}")
    print("[plot-kappa] overlaid plots:")
    for output in result["plots"]:
        print(f"  {output}")
    return 0


def _plot_dos(args: argparse.Namespace, config: dict) -> int:
    from .dos_plot import mesh_tag, plot_dos_comparison
    from .stages import plot_dos_archive

    mlp_names = list(dict.fromkeys(args.mlp or [active_mlp(config)]))
    results = []
    for mlp_name in mlp_names:
        mlp_config = (
            config
            if active_mlp(config) == mlp_name
            else load_config(args.config, mlp_override=mlp_name)
        )
        result = plot_dos_archive(
            mlp_config,
            mesh=args.mesh,
            method=args.method,
            sigma=args.sigma,
            frequency_min=args.frequency_min,
            frequency_max=args.frequency_max,
            frequency_pitch=args.frequency_pitch,
            dpi=args.dpi,
        )
        results.append(result)
        print(f"[plot-dos] {mlp_name} outputs:")
        for key in ("plot", "data", "metadata"):
            print(f"  {result[key]}")

    if len(results) > 1:
        comparison_root = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else plot_archive_root(config) / "plots" / "dos"
        )
        comparison = plot_dos_comparison(
            results,
            output_dir=comparison_root / mesh_tag(results[0]["mesh"]),
            dpi=int(args.dpi or config.get("dos", {}).get("dpi", 200)),
        )
        print("[plot-dos] comparison outputs:")
        for key in ("plot", "data", "metadata"):
            print(f"  {comparison[key]}")
    return 0


def _maybe_relaunch_plot_dos(
    config: dict, args: argparse.Namespace
) -> int | None:
    if args.no_relaunch or not bool(config["execution"].get("auto_relaunch", True)):
        return None

    target_env = conda_env_for_stage(config, "ph3-fc")
    if not target_env or os.environ.get("CONDA_DEFAULT_ENV") == target_env:
        return None

    conda_base = _conda_base()
    conda_sh = Path(conda_base) / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.exists():
        raise ConfigError(f"conda.sh not found: {conda_sh}")

    command = [
        "python",
        "-m",
        "mlp_phonon_workflow",
        "plot-dos",
        "--config",
        str(Path(args.config).resolve()),
        "--no-relaunch",
    ]
    if args.mlp:
        command.extend(["--mlp", *args.mlp])
    if args.output_dir:
        command.extend(["--output-dir", str(Path(args.output_dir).resolve())])
    if args.mesh:
        command.extend(["--mesh", *(str(value) for value in args.mesh)])
    for option, value in (
        ("--method", args.method),
        ("--sigma", args.sigma),
        ("--frequency-min", args.frequency_min),
        ("--frequency-max", args.frequency_max),
        ("--frequency-pitch", args.frequency_pitch),
        ("--dpi", args.dpi),
    ):
        if value is not None:
            command.extend([option, str(value)])

    shell_command = " && ".join(
        [
            f"source {shlex.quote(str(conda_sh))}",
            f"conda activate {shlex.quote(target_env)}",
            " ".join(shlex.quote(part) for part in command),
        ]
    )
    env = os.environ.copy()
    env.update(extra_env(config))
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[plot-dos] relaunch in conda env '{target_env}'", flush=True)
    completed = subprocess.run(["bash", "-lc", shell_command], cwd=repo_root, env=env)
    return int(completed.returncode)
