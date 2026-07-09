from __future__ import annotations

import json
import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when the workflow configuration is invalid."""


DEFAULT_CONFIG: dict[str, Any] = {
    "workflow": {
        "active_mlp": "mattersim",
        "run_dir": "runs/{mlp}",
        "input_poscar": "POSCAR",
        "overwrite_next_inputs": False,
    },
    "execution": {
        "auto_relaunch": True,
        "phonopy_conda_env": "",
        "extra_env": {
            "MPLCONFIGDIR": "{run_dir}/.matplotlib",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        },
    },
    "mlp": {
        "mattersim": {
            "calculator": "mattersim",
            "conda_env": "mattersim_env",
            "kwargs": {
                "device": "cuda",
                "compute_stress": True,
            },
        },
        "mace_mp": {
            "calculator": "mace_mp",
            "conda_env": "mace_env",
            "kwargs": {
                "model": "medium",
                "device": "cuda",
                "default_dtype": "float64",
                "dispersion": False,
            },
        },
        "sevennet": {
            "calculator": "sevennet",
            "conda_env": "sevenn_env",
            "kwargs": {
                "model": "7net-0",
                "device": "auto",
            },
        },
    },
    "relax": {
        "optimizer": "LBFGS",
        "fmax": 0.01,
        "max_steps": 500,
        "write_extxyz": True,
    },
    "phonopy": {
        "calculator": "vasp",
        "supercell_matrix": [2, 2, 2],
        "primitive_matrix": "auto",
        "symprec": 1.0e-5,
        "is_symmetry": True,
        "use_SNF_supercell": False,
    },
    "displacements": {
        "distance": 0.01,
        "is_plusminus": "auto",
        "is_diagonal": True,
        "is_trigonal": False,
        "zfill_width": 3,
        "supercell_filename_prefix": "POSCAR",
    },
    "forces": {
        "subtract_drift": True,
        "write_npz": True,
    },
    "band": {
        "auto": True,
        "nqpoints": 101,
        "with_eigenvectors": False,
        "with_group_velocities": False,
        "is_band_connection": False,
        "fc_calculator": "traditional",
        "fc_calculator_options": "",
        "write_force_constants": False,
        "force_constants_format": "hdf5",
        "paths": [],
        "labels": [],
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | os.PathLike[str],
    *,
    mlp_override: str | None = None,
    poscar_override: str | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    user_config = _read_config_file(config_path)
    config = deep_merge(DEFAULT_CONFIG, user_config)
    if mlp_override:
        config["workflow"]["active_mlp"] = mlp_override
    if poscar_override:
        config["workflow"]["input_poscar"] = poscar_override

    config["_meta"] = {
        "config_path": config_path,
        "config_dir": config_path.parent,
    }
    validate_config(config)
    return config


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        if suffix == ".toml":
            return tomllib.load(handle)
        if suffix == ".json":
            return json.load(handle)
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigError(
                    "YAML config requires PyYAML. Use TOML or run from an env with PyYAML."
                ) from exc
            return yaml.safe_load(handle) or {}
    raise ConfigError(f"Unsupported config format: {path.suffix}")


def validate_config(config: dict[str, Any]) -> None:
    mlp_name = active_mlp(config)
    mlp_table = config.get("mlp", {})
    if mlp_name not in mlp_table:
        choices = ", ".join(sorted(mlp_table))
        raise ConfigError(f"Unknown active_mlp '{mlp_name}'. Choices: {choices}")
    calculator = mlp_table[mlp_name].get("calculator")
    if calculator not in {"mattersim", "mace_mp", "sevennet"}:
        raise ConfigError(
            f"Unsupported calculator '{calculator}' for mlp.{mlp_name}."
        )


def active_mlp(config: dict[str, Any]) -> str:
    return str(config["workflow"]["active_mlp"])


def context(config: dict[str, Any]) -> dict[str, str]:
    mlp_name = active_mlp(config)
    run_dir_text = str(_expand_text(config["workflow"]["run_dir"], {"mlp": mlp_name}))
    run_dir = _as_abs_path(run_dir_text, config["_meta"]["config_dir"])
    return {
        "mlp": mlp_name,
        "run_dir": str(run_dir),
        "config_dir": str(config["_meta"]["config_dir"]),
    }


def expand_text(value: str, config: dict[str, Any]) -> str:
    return _expand_text(value, context(config))


def _expand_text(value: str, values: dict[str, str]) -> str:
    try:
        return value.format(**values)
    except KeyError as exc:
        raise ConfigError(f"Unknown format key in config string '{value}': {exc}") from exc


def run_dir(config: dict[str, Any]) -> Path:
    return Path(context(config)["run_dir"])


def input_poscar(config: dict[str, Any]) -> Path:
    raw = expand_text(str(config["workflow"]["input_poscar"]), config)
    return _as_abs_path(raw, config["_meta"]["config_dir"])


def conda_env_for_stage(config: dict[str, Any], stage: str) -> str:
    mlp_name = active_mlp(config)
    mlp_env = str(config["mlp"][mlp_name].get("conda_env", ""))
    if stage in {"relax", "forces"}:
        return mlp_env
    phonopy_env = str(config["execution"].get("phonopy_conda_env", ""))
    return phonopy_env or mlp_env


def extra_env(config: dict[str, Any]) -> dict[str, str]:
    raw_env = config["execution"].get("extra_env", {})
    resolved: dict[str, str] = {}
    for key, value in raw_env.items():
        resolved[str(key)] = expand_text(str(value), config)
    return resolved


def _as_abs_path(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
