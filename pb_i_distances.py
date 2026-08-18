#!/usr/bin/env python3
"""POSCAR에서 주기경계조건을 고려한 인접 Pb-I 거리를 출력한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import primitive_neighbor_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "POSCAR를 읽고 주기경계조건을 고려하여 컷오프 이내의 "
            "Pb-I 거리를 출력합니다."
        )
    )
    parser.add_argument("poscar", type=Path, help="입력 POSCAR 파일 경로")
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=4.0,
        help="Pb-I 인접 거리 컷오프(Å, 기본값: 4.0)",
    )
    return parser.parse_args()


def find_pb_i_neighbors(
    poscar: Path, cutoff: float
) -> list[tuple[int, int, np.ndarray, float]]:
    """컷오프 이내의 Pb 원자와 I 원자(주기 이미지 포함)를 찾는다."""
    if cutoff <= 0:
        raise ValueError("cutoff은 0보다 커야 합니다.")
    if not poscar.is_file():
        raise FileNotFoundError(f"POSCAR 파일을 찾을 수 없습니다: {poscar}")

    atoms = read(poscar, format="vasp")
    symbols = np.asarray(atoms.get_chemical_symbols())

    if not np.any(symbols == "Pb"):
        raise ValueError("구조에 Pb 원자가 없습니다.")
    if not np.any(symbols == "I"):
        raise ValueError("구조에 I 원자가 없습니다.")

    atom_i, atom_j, shifts, distances = primitive_neighbor_list(
        "ijSd",
        atoms.pbc,
        atoms.cell,
        atoms.positions,
        cutoff,
        self_interaction=False,
    )

    pb_i_mask = (symbols[atom_i] == "Pb") & (symbols[atom_j] == "I")
    neighbors = [
        (int(i), int(j), np.asarray(shift, dtype=int), float(distance))
        for i, j, shift, distance in zip(
            atom_i[pb_i_mask],
            atom_j[pb_i_mask],
            shifts[pb_i_mask],
            distances[pb_i_mask],
            strict=True,
        )
    ]
    return sorted(
        neighbors,
        key=lambda item: (item[0], item[3], item[1], tuple(item[2])),
    )


def main() -> None:
    args = parse_args()

    try:
        neighbors = find_pb_i_neighbors(args.poscar, args.cutoff)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"오류: {error}") from error

    print(f"POSCAR: {args.poscar.resolve()}")
    print(f"인접 거리 컷오프: {args.cutoff:.3f} Å")

    if not neighbors:
        print("컷오프 이내의 Pb-I 쌍이 없습니다.")
        return

    print("\n Pb 번호   I 번호   I 주기 이미지 (a,b,c)    거리 (Å)")
    print("------------------------------------------------------")
    for pb_index, i_index, shift, distance in neighbors:
        shift_text = f"({shift[0]:+d},{shift[1]:+d},{shift[2]:+d})"
        print(
            f"{pb_index + 1:8d} {i_index + 1:8d}"
            f"   {shift_text:>18s}   {distance:10.6f}"
        )

    print(f"\n총 {len(neighbors)}개의 인접 Pb-I 결합")


if __name__ == "__main__":
    main()
