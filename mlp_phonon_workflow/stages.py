from __future__ import annotations

from typing import Any

from .phono3py_stages import (
    _run_ph3_displace,
    _run_ph3_fc,
    _run_ph3_forces,
    _run_thermal_conductivity,
)
from .phonopy_stages import _run_band, _run_displace, _run_forces, _run_relax
from .stage_archive import (
    _archive_band_outputs,
    _archive_dos_inputs,
    _archive_kappa_outputs,
    _plot_kappa_outputs,
    plot_dos_archive,
)
from .stage_common import (
    EXPECTED_OUTPUTS,
    PHONO3PY_ONLY_STAGES,
    PHONOPY_STAGES,
    STAGE_DIR_NAMES,
    STAGES,
    StagePaths,
    stage_paths,
    stage_status,
)

_STAGE_RUNNERS = {
    "relax": _run_relax,
    "displace": _run_displace,
    "forces": _run_forces,
    "band": _run_band,
    "ph3-displace": _run_ph3_displace,
    "ph3-forces": _run_ph3_forces,
    "ph3-fc": _run_ph3_fc,
}


def run_stage(config: dict[str, Any], stage: str, *, force: bool = False) -> None:
    """Create the stage tree and run one configured workflow stage."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")

    for name in STAGES:
        paths = stage_paths(config, name)
        paths.input.mkdir(parents=True, exist_ok=True)
        paths.output.mkdir(parents=True, exist_ok=True)

    if stage.startswith("kappa-"):
        method = "rta" if stage == "kappa-rta" else "iterative"
        _run_thermal_conductivity(config, method=method, force=force)
        return

    _STAGE_RUNNERS[stage](config, force=force)
