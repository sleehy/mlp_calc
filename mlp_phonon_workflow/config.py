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
    "plot_archive": {
        "enabled": True,
        "root": "plot_archive",
        "overwrite": True,
    },
    "dos": {
        "enabled": True,
        "mesh": [40, 40, 40],
        "method": "tetrahedron",
        "sigma": None,
        "frequency_min": None,
        "frequency_max": None,
        "frequency_pitch": None,
        "is_gamma_center": True,
        "is_mesh_symmetry": True,
        "dpi": 200,
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
    "phono3py": {
        "calculator": "vasp",
        "supercell_matrix": [2, 2, 2],
        "phonon_supercell_matrix": [2, 2, 2],
        "primitive_matrix": "auto",
        "symprec": 1.0e-5,
        "is_symmetry": True,
        "is_mesh_symmetry": True,
        "use_grg": False,
        "make_r0_average": True,
        "cutoff_frequency": 1.0e-4,
        "log_level": 1,
        "lang": "Rust",
        "displacements": {
            "zfill_width": 5,
            "fc3": {
                "distance": 0.03,
                "is_plusminus": "auto",
                "is_diagonal": True,
            },
            "fc2": {
                "distance": 0.03,
                "is_plusminus": "auto",
                "is_diagonal": False,
            },
        },
        "forces": {
            "subtract_drift": True,
            "write_energies": True,
        },
        "force_constants": {
            "fc_calculator": "traditional",
            "fc_calculator_options": "",
            "symmetrize_fc2": True,
            "symmetrize_fc3": True,
            "is_compact_fc": True,
            "use_symfc_projector": False,
            "compression": "gzip",
        },
    },
    "thermal_conductivity": {
        "methods": ["rta", "iterative"],
        "mesh": [8, 8, 8],
        "temperatures": [300.0],
        "sigmas": [],
        "sigma_cutoff": None,
        "is_isotope": False,
        "mass_variances": [],
        "boundary_mfp": None,
        "is_kappa_star": True,
        "gv_delta_q": None,
        "is_full_pp": False,
        "transport_type": None,
        "compression": "gzip",
        "frequency_scale_factor": None,
        "constant_averaged_interaction": None,
        "nac_q_direction": [],
        "symmetrize_fc3q": False,
        "lapack_zheev_uplo": None,
        "openmp_per_triplets": None,
        "grid_points": [],
        "solve_collective_phonon": False,
        "rta": {
            "use_ave_pp": False,
            "write_gamma": False,
            "read_gamma": False,
            "is_N_U": False,
            "write_gamma_detail": False,
            "write_pp": False,
            "read_pp": False,
        },
        "iterative": {
            "is_reducible_collision_matrix": False,
            "pinv_cutoff": 1.0e-8,
            "pinv_method": 0,
            "pinv_solver": 0,
            "write_collision": False,
            "read_collision": False,
            "write_pp": False,
            "read_pp": False,
            "write_LBTE_solution": False,
        },
        "plots": {
            "enabled": True,
            "bins": 200,
            "mfp_unit": "nm",
            "dpi": 200,
            "temperatures": [],
        },
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

    methods = config.get("thermal_conductivity", {}).get("methods", [])
    if not isinstance(methods, list):
        raise ConfigError("thermal_conductivity.methods must be a list.")
    invalid_methods = sorted(set(methods) - {"rta", "iterative"})
    if invalid_methods:
        raise ConfigError(
            "thermal_conductivity.methods only accepts 'rta' and 'iterative': "
            + ", ".join(invalid_methods)
        )

    mesh = config.get("thermal_conductivity", {}).get("mesh", [])
    if not isinstance(mesh, list) or len(mesh) != 3:
        raise ConfigError(
            "thermal_conductivity.mesh must contain 3 mesh numbers or three "
            "rows of a 3x3 grid matrix."
        )
    if any(isinstance(row, list) for row in mesh) and not all(
        isinstance(row, list) and len(row) == 3 for row in mesh
    ):
        raise ConfigError(
            "thermal_conductivity.mesh must be either 3 numbers or a complete "
            "3x3 matrix."
        )

    plot_config = config.get("thermal_conductivity", {}).get("plots", {})
    mfp_unit = str(plot_config.get("mfp_unit", "nm"))
    if mfp_unit not in {"angstrom", "nm", "um"}:
        raise ConfigError(
            "thermal_conductivity.plots.mfp_unit must be 'angstrom', 'nm', or 'um'."
        )
    if int(plot_config.get("bins", 200)) < 2:
        raise ConfigError("thermal_conductivity.plots.bins must be at least 2.")

    archive_config = config.get("plot_archive", {})
    if bool(archive_config.get("enabled", True)):
        archive_root = str(archive_config.get("root", "plot_archive")).strip()
        if not archive_root:
            raise ConfigError("plot_archive.root must not be empty when enabled.")

    dos_config = config.get("dos", {})
    dos_mesh = dos_config.get("mesh", [40, 40, 40])
    if (
        not isinstance(dos_mesh, list)
        or len(dos_mesh) != 3
        or any(int(value) <= 0 for value in dos_mesh)
    ):
        raise ConfigError("dos.mesh must contain three positive integers.")
    dos_method = str(dos_config.get("method", "tetrahedron")).lower()
    if dos_method not in {"tetrahedron", "gaussian"}:
        raise ConfigError("dos.method must be 'tetrahedron' or 'gaussian'.")
    dos_sigma = dos_config.get("sigma")
    if dos_method == "gaussian" and (
        dos_sigma is None or float(dos_sigma) <= 0
    ):
        raise ConfigError("dos.sigma must be positive when dos.method='gaussian'.")


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


def plot_archive_root(config: dict[str, Any]) -> Path:
    raw = expand_text(str(config["plot_archive"].get("root", "plot_archive")), config)
    return _as_abs_path(raw, config["_meta"]["config_dir"])


def input_poscar(config: dict[str, Any]) -> Path:
    raw = expand_text(str(config["workflow"]["input_poscar"]), config)
    return _as_abs_path(raw, config["_meta"]["config_dir"])


def conda_env_for_stage(config: dict[str, Any], stage: str) -> str:
    mlp_name = active_mlp(config)
    mlp_env = str(config["mlp"][mlp_name].get("conda_env", ""))
    if stage in {
        "relax",
        "forces",
        "ph3-displace",
        "ph3-forces",
        "ph3-fc",
        "kappa-rta",
        "kappa-iterative",
    }:
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
