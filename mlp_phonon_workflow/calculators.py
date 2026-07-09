from __future__ import annotations

from typing import Any


def build_calculator(config: dict[str, Any]):
    """Build the ASE calculator selected by config.workflow.active_mlp."""
    mlp_name = config["workflow"]["active_mlp"]
    mlp_config = config["mlp"][mlp_name]
    calculator_name = mlp_config["calculator"]
    kwargs = dict(mlp_config.get("kwargs", {}))

    if calculator_name == "mattersim":
        from mattersim.forcefield.potential import MatterSimCalculator

        return MatterSimCalculator(**kwargs)

    if calculator_name == "mace_mp":
        from mace.calculators import mace_mp

        return mace_mp(**kwargs)

    if calculator_name == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        _validate_sevennet_kwargs(kwargs)
        return SevenNetCalculator(**kwargs)

    raise ValueError(f"Unsupported calculator: {calculator_name}")


def _validate_sevennet_kwargs(kwargs: dict[str, Any]) -> None:
    device = str(kwargs.get("device", "auto")).lower()
    modal_names = {
        "mpa",
        "omat24",
        "matpes_pbe",
        "matpes_r2scan",
        "mp_r2scan",
        "oc20",
        "oc22",
        "odac23",
        "omol25_low",
        "omol25_high",
        "spice",
        "qcml",
        "pet_mad",
    }
    if device in modal_names:
        raise ValueError(
            "SevenNet config has device="
            f"{kwargs.get('device')!r}, but this is a modal name. "
            "Use device='auto' or 'cuda' or 'cpu', and set modal="
            f"{kwargs.get('device')!r}."
        )
