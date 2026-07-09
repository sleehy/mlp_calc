from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .config import ConfigError, conda_env_for_stage, extra_env, load_config, run_dir
from .stages import STAGES, run_stage, stage_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlp-phonon-workflow",
        description="Resumable POSCAR -> MLP relax -> phonopy -> FORCE_SETS -> band.yaml workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one stage or all stages.")
    run_parser.add_argument(
        "stage",
        choices=[*STAGES, "all"],
        help="Stage to run. Use 'all' to run relax, displace, forces, band.",
    )
    _add_common_args(run_parser)
    run_parser.add_argument("--force", action="store_true", help="Rerun even if output exists.")
    run_parser.add_argument(
        "--no-relaunch",
        action="store_true",
        help="Do not auto-relaunch into the configured conda env.",
    )

    status_parser = subparsers.add_parser("status", help="Show resumable stage status.")
    _add_common_args(status_parser)

    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, mlp_override=args.mlp, poscar_override=args.poscar)
        _apply_extra_env(config)
        if args.command == "status":
            _print_status(config)
            return 0

        stages = STAGES if args.stage == "all" else (args.stage,)
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


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        default="config.toml",
        help="Path to TOML, YAML, or JSON config. Default: config.toml",
    )
    parser.add_argument(
        "--mlp",
        choices=["mattersim", "mace_mp", "sevennet"],
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
        print(f"{row['stage']:9s} {row['status']:8s} {row['output']}")
