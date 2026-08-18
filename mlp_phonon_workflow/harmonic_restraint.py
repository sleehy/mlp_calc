from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic


class HarmonicBondRestraintCalculator(Calculator):
    """Harmonic bond restraint energy, forces, and virial stress."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        pairs: Sequence[tuple[int, int]],
        target_lengths: Sequence[float],
        spring_constant: float,
    ) -> None:
        super().__init__()
        self.pairs = tuple((int(a), int(b)) for a, b in pairs)
        self.target_lengths = np.asarray(target_lengths, dtype=float)
        self.spring_constant = float(spring_constant)

    def calculate(
        self,
        atoms: Any = None,
        properties: Sequence[str] = ("energy", "forces"),
        system_changes: Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise RuntimeError("Harmonic restraint requires an Atoms object.")

        energy = 0.0
        forces = np.zeros((len(atoms), 3), dtype=float)
        virial = np.zeros((3, 3), dtype=float)

        for (atom_a, atom_b), target_length in zip(
            self.pairs,
            self.target_lengths,
            strict=True,
        ):
            displacement = atoms.positions[atom_b] - atoms.positions[atom_a]
            displacement, distance = find_mic(
                displacement,
                atoms.cell,
                atoms.pbc,
            )
            distance = float(distance)
            if distance == 0.0:
                raise RuntimeError(
                    "Cannot apply a bond restraint to coincident atoms "
                    f"{atom_a} and {atom_b}."
                )

            extension = distance - float(target_length)
            energy += 0.5 * self.spring_constant * extension**2
            force_on_a = (
                self.spring_constant * extension * displacement / distance
            )
            forces[atom_a] += force_on_a
            forces[atom_b] -= force_on_a
            virial += np.outer(force_on_a, displacement)

        volume = float(atoms.get_volume())
        stress = 0.5 * (virial + virial.T) / volume
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
            "stress": stress,
        }
