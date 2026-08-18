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

        if args.command == "plot-band-dos":
            config = load_config(args.config, mlp_override=args.mlp)
            _apply_extra_env(config)
            status = _maybe_relaunch_plot_band_dos(config, args)
            if status is not None:
                return status
            return _plot_band_dos(args, config)

        if args.command == "plot-band":
            config = load_config(args.config, mlp_override=args.mlp)
            _apply_extra_env(config)
            status = _maybe_relaunch_plot_band(config, args)
            if status is not None:
                return status
            return _plot_band(args, config)

        config = load_config(
            args.config,
            mlp_override=args.mlp,
            poscar_override=args.poscar,
            run_dir_override=args.run_dir,
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
    dos_parser.add_argument(
        "--projected",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Generate element-projected DOS (or disable it with --no-projected). "
            "Default: dos.projected from the config."
        ),
    )
    dos_parser.add_argument(
        "--phonopy-yaml",
        help=(
            "Harmonic phonopy_params.yaml input. When set, use the phonopy "
            "result instead of the archived phono3py fc2 inputs."
        ),
    )
    dos_parser.add_argument(
        "--force-constants",
        help=(
            "Optional FORCE_CONSTANTS or force_constants.hdf5 used with "
            "--phonopy-yaml. Otherwise force constants are reconstructed from "
            "forces stored in the YAML."
        ),
    )
    dos_parser.add_argument(
        "--born",
        help="Optional BORN file used with --phonopy-yaml.",
    )
    dos_parser.add_argument("--dpi", type=int, help="Plot DPI. Default: dos.dpi.")
    _add_config_arg(dos_parser)
    dos_parser.add_argument(
        "--mlp",
        nargs="+",
        choices=MLP_CHOICES,
        help=(
            "MLP environment/source label. Without --phonopy-yaml, one or more "
            "MLP archives may be selected and overlaid. Default: "
            "workflow.active_mlp."
        ),
    )
    dos_parser.add_argument(
        "-o",
        "--output-dir",
        help=(
            "Output directory for --phonopy-yaml, or multi-MLP comparison "
            "output root. Defaults beside the phonopy input or under "
            "<plot_archive>/plots/dos/."
        ),
    )
    dos_parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not relaunch into the selected MLP conda environment.",
    )

    band_dos_parser = subparsers.add_parser(
        "plot-band-dos",
        help="Plot a phonopy band structure and element-projected DOS together.",
    )
    band_dos_parser.add_argument(
        "--phonopy-yaml",
        required=True,
        help="Harmonic phonopy_params.yaml containing forces or force constants.",
    )
    band_dos_parser.add_argument(
        "--force-constants",
        help="Optional FORCE_CONSTANTS or force_constants.hdf5.",
    )
    band_dos_parser.add_argument("--born", help="Optional BORN file for NAC.")
    band_dos_parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Projected-DOS q-point mesh. Default: dos.mesh from the config.",
    )
    band_dos_parser.add_argument(
        "--method",
        choices=["tetrahedron", "gaussian"],
        help="DOS integration method. Default: dos.method from the config.",
    )
    band_dos_parser.add_argument(
        "--sigma",
        type=float,
        help="Gaussian width in THz; required for method=gaussian.",
    )
    band_dos_parser.add_argument("--frequency-min", type=float)
    band_dos_parser.add_argument("--frequency-max", type=float)
    band_dos_parser.add_argument("--frequency-pitch", type=float)
    band_dos_parser.add_argument(
        "--customizer",
        help=(
            "Python file defining customize(fig). It is called after Phonopy "
            "creates the Matplotlib Figure and before the PNG is saved."
        ),
    )
    band_dos_parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output PNG. Default: phonon_band_projected_dos.png beside the "
            "phonopy YAML."
        ),
    )
    band_dos_parser.add_argument("--dpi", type=int, help="Default: dos.dpi.")
    _add_config_arg(band_dos_parser)
    band_dos_parser.add_argument(
        "--mlp",
        choices=MLP_CHOICES,
        help="MLP environment/source selection. Default: workflow.active_mlp.",
    )
    band_dos_parser.add_argument(
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

    band_plot_parser = subparsers.add_parser(
        "plot-band",
        help="Plot a phonopy dispersion relation, optionally with experimental data.",
    )
    band_plot_parser.add_argument(
        "--mode",
        choices=["dispersion", "experiment"],
        default="dispersion",
        help=(
            "Plot only the calculated dispersion (default), or overlay experimental "
            "CSV points."
        ),
    )
    band_plot_parser.add_argument(
        "--band-yaml",
        help=(
            "Input band.yaml. Default: the active MLP's archived "
            "band/inputs/band.yaml."
        ),
    )
    band_plot_parser.add_argument(
        "--experiment-csv",
        help="CSV containing experimental x,y points; required for mode=experiment.",
    )
    band_plot_parser.add_argument(
        "--high-symmetry-positions",
        nargs="+",
        type=float,
        help=(
            "Experimental x positions of every band-path boundary. By default, "
            "calculated cumulative distances are normalized to 0--1."
        ),
    )
    band_plot_parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Default: next to the archived input under plots/.",
    )
    band_plot_parser.add_argument("--dpi", type=int, default=200, help="Plot DPI.")
    _add_config_arg(band_plot_parser)
    band_plot_parser.add_argument(
        "--mlp",
        choices=MLP_CHOICES,
        help="Select the MLP archive. Default: workflow.active_mlp.",
    )
    band_plot_parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not relaunch into the selected MLP conda environment.",
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
    parser.add_argument(
        "--run-dir",
        help="Override workflow.run_dir for this command.",
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
    if args.run_dir:
        cmd.extend(["--run-dir", args.run_dir])

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
    from .dos_plot import mesh_tag, plot_dos_comparison, plot_dos_from_phonopy
    from .stages import plot_dos_archive

    if args.phonopy_yaml:
        if args.mlp and len(args.mlp) > 1:
            raise ConfigError(
                "--phonopy-yaml accepts at most one --mlp value because it "
                "describes a single harmonic calculation."
            )
        dos_settings = config.get("dos", {})
        mesh = args.mesh or dos_settings.get("mesh", [40, 40, 40])
        source = Path(args.phonopy_yaml).expanduser().resolve()
        destination = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else source.parent / "dos" / mesh_tag(mesh)
        )
        phonopy_config = config["phonopy"]
        band_config = config["band"]
        result = plot_dos_from_phonopy(
            source,
            mesh=mesh,
            output_dir=destination,
            source_label=active_mlp(config),
            force_constants_file=args.force_constants,
            calculator=str(phonopy_config.get("calculator", "vasp")),
            born_file=args.born,
            fc_calculator=(
                str(band_config.get("fc_calculator", "traditional")).strip() or None
            ),
            fc_calculator_options=(
                str(band_config.get("fc_calculator_options", "")).strip() or None
            ),
            method=str(args.method or dos_settings.get("method", "tetrahedron")),
            sigma=args.sigma if args.sigma is not None else dos_settings.get("sigma"),
            frequency_min=(
                args.frequency_min
                if args.frequency_min is not None
                else dos_settings.get("frequency_min")
            ),
            frequency_max=(
                args.frequency_max
                if args.frequency_max is not None
                else dos_settings.get("frequency_max")
            ),
            frequency_pitch=(
                args.frequency_pitch
                if args.frequency_pitch is not None
                else dos_settings.get("frequency_pitch")
            ),
            is_gamma_center=bool(dos_settings.get("is_gamma_center", True)),
            is_mesh_symmetry=bool(dos_settings.get("is_mesh_symmetry", True)),
            projected=(
                bool(args.projected)
                if args.projected is not None
                else bool(dos_settings.get("projected", False))
            ),
            symprec=float(phonopy_config.get("symprec", 1.0e-5)),
            dpi=int(args.dpi or dos_settings.get("dpi", 200)),
        )
        print(f"[plot-dos] phonopy input: {source}")
        print("[plot-dos] outputs:")
        _print_dos_outputs(result)
        return 0

    if args.force_constants or args.born:
        raise ConfigError(
            "--force-constants and --born require --phonopy-yaml."
        )

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
            projected=args.projected,
            dpi=args.dpi,
        )
        results.append(result)
        print(f"[plot-dos] {mlp_name} outputs:")
        _print_dos_outputs(result)

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


def _print_dos_outputs(result: dict) -> None:
    for key in ("plot", "data", "projected_plot", "projected_data", "metadata"):
        if result.get(key):
            print(f"  {result[key]}")


def _plot_band_dos(args: argparse.Namespace, config: dict) -> int:
    from .dos_plot import save_band_with_projected_dos_from_phonopy

    dos_config = config.get("dos", {})
    band_config = config["band"]
    phonopy_config = config["phonopy"]
    result = save_band_with_projected_dos_from_phonopy(
        args.phonopy_yaml,
        mesh=args.mesh or dos_config.get("mesh", [40, 40, 40]),
        output_file=args.output,
        force_constants_file=args.force_constants,
        calculator=str(phonopy_config.get("calculator", "vasp")),
        born_file=args.born,
        fc_calculator=(
            str(band_config.get("fc_calculator", "traditional")).strip() or None
        ),
        fc_calculator_options=(
            str(band_config.get("fc_calculator_options", "")).strip() or None
        ),
        band_auto=bool(band_config.get("auto", True)),
        band_nqpoints=int(band_config.get("nqpoints", 101)),
        band_paths=band_config.get("paths", []),
        band_labels=band_config.get("labels", []),
        is_band_connection=bool(band_config.get("is_band_connection", False)),
        method=str(args.method or dos_config.get("method", "tetrahedron")),
        sigma=args.sigma if args.sigma is not None else dos_config.get("sigma"),
        frequency_min=(
            args.frequency_min
            if args.frequency_min is not None
            else dos_config.get("frequency_min")
        ),
        frequency_max=(
            args.frequency_max
            if args.frequency_max is not None
            else dos_config.get("frequency_max")
        ),
        frequency_pitch=(
            args.frequency_pitch
            if args.frequency_pitch is not None
            else dos_config.get("frequency_pitch")
        ),
        is_gamma_center=bool(dos_config.get("is_gamma_center", True)),
        symprec=float(phonopy_config.get("symprec", 1.0e-5)),
        customizer_file=args.customizer,
        dpi=int(args.dpi or dos_config.get("dpi", 200)),
    )
    print(f"[plot-band-dos] input: {result['phonopy_yaml']}")
    print(f"[plot-band-dos] plot: {result['plot']}")
    print(f"[plot-band-dos] metadata: {result['metadata']}")
    return 0


def _plot_band(args: argparse.Namespace, config: dict) -> int:
    from .kappa_plot import plot_band_dispersion, plot_band_with_experiment

    source = (
        Path(args.band_yaml).expanduser().resolve()
        if args.band_yaml
        else (
            plot_archive_root(config)
            / active_mlp(config)
            / "band"
            / "inputs"
            / "band.yaml"
        )
    )
    output = Path(args.output).expanduser().resolve() if args.output else None
    if args.mode == "experiment":
        if not args.experiment_csv:
            raise ConfigError(
                "plot-band --mode experiment requires --experiment-csv."
            )
        result = plot_band_with_experiment(
            source,
            args.experiment_csv,
            output_file=output,
            high_symmetry_positions=args.high_symmetry_positions,
            dpi=args.dpi,
        )
    else:
        if args.experiment_csv or args.high_symmetry_positions:
            raise ConfigError(
                "--experiment-csv and --high-symmetry-positions require "
                "plot-band --mode experiment."
            )
        result = plot_band_dispersion(
            source,
            output_file=output,
            dpi=args.dpi,
        )
    print(f"[plot-band] mode: {result['mode']}")
    print(f"[plot-band] input: {result['source']}")
    print(f"[plot-band] output: {result['plot']}")
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
    if args.phonopy_yaml:
        command.extend(
            ["--phonopy-yaml", str(Path(args.phonopy_yaml).expanduser().resolve())]
        )
    if args.force_constants:
        command.extend(
            [
                "--force-constants",
                str(Path(args.force_constants).expanduser().resolve()),
            ]
        )
    if args.born:
        command.extend(["--born", str(Path(args.born).expanduser().resolve())])
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
    if args.projected is not None:
        command.append("--projected" if args.projected else "--no-projected")

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


def _maybe_relaunch_plot_band(
    config: dict, args: argparse.Namespace
) -> int | None:
    if args.no_relaunch or not bool(config["execution"].get("auto_relaunch", True)):
        return None

    target_env = conda_env_for_stage(config, "band")
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
        "plot-band",
        "--config",
        str(Path(args.config).resolve()),
        "--mode",
        args.mode,
        "--dpi",
        str(args.dpi),
        "--no-relaunch",
    ]
    if args.mlp:
        command.extend(["--mlp", args.mlp])
    if args.band_yaml:
        command.extend(["--band-yaml", str(Path(args.band_yaml).resolve())])
    if args.experiment_csv:
        command.extend(
            ["--experiment-csv", str(Path(args.experiment_csv).resolve())]
        )
    if args.high_symmetry_positions:
        command.extend(
            [
                "--high-symmetry-positions",
                *(str(value) for value in args.high_symmetry_positions),
            ]
        )
    if args.output:
        command.extend(["--output", str(Path(args.output).resolve())])

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
    print(f"[plot-band] relaunch in conda env '{target_env}'", flush=True)
    completed = subprocess.run(["bash", "-lc", shell_command], cwd=repo_root, env=env)
    return int(completed.returncode)


def _maybe_relaunch_plot_band_dos(
    config: dict, args: argparse.Namespace
) -> int | None:
    if args.no_relaunch or not bool(config["execution"].get("auto_relaunch", True)):
        return None

    target_env = conda_env_for_stage(config, "band")
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
        "plot-band-dos",
        "--config",
        str(Path(args.config).resolve()),
        "--phonopy-yaml",
        str(Path(args.phonopy_yaml).expanduser().resolve()),
        "--no-relaunch",
    ]
    if args.mlp:
        command.extend(["--mlp", args.mlp])
    if args.force_constants:
        command.extend(
            [
                "--force-constants",
                str(Path(args.force_constants).expanduser().resolve()),
            ]
        )
    if args.born:
        command.extend(["--born", str(Path(args.born).expanduser().resolve())])
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
    if args.output:
        command.extend(["--output", str(Path(args.output).expanduser().resolve())])
    if args.customizer:
        command.extend(
            ["--customizer", str(Path(args.customizer).expanduser().resolve())]
        )

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
    print(f"[plot-band-dos] relaunch in conda env '{target_env}'", flush=True)
    completed = subprocess.run(["bash", "-lc", shell_command], cwd=repo_root, env=env)
    return int(completed.returncode)
