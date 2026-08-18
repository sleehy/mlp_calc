from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_POSCAR = SCRIPT_DIR.parent / "t-FAPb" / "POSCAR"

# N1 -> N2 벡터가 향할 결정 방향
NN_UVW = np.array([1, 0, 0])
NN_TAG = "".join(str(component) for component in NN_UVW)
OUTPUT_POSCAR = SCRIPT_DIR.parent / "t-FAPb" / f"POSCAR_FA_NN_{NN_TAG}"

# 분율좌표로 나타낸 회전 중심
ROTATION_CENTER_SCALED = np.array([0.5, 0.5, 0.5])

# POSCAR에서 FA 분자를 구성하는 원자 번호 (Python 번호: 0부터 시작)
fa_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # 반드시 확인

# N1 -> N2 방향을 정의하는 두 N 원자의 번호
n1_index = 1  # 반드시 확인
n2_index = 2  # 반드시 확인


def lattice_direction(uvw, cell):
    """결정 방향 [uvw]를 정규화된 Cartesian 벡터로 변환합니다."""
    vector = np.asarray(uvw, dtype=float) @ cell
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("NN_UVW must not be [0, 0, 0].")
    return vector / norm


def rotation_matrix_from_vectors(source, target):
    """source를 target에 맞추는 최소 회전행렬을 반환합니다."""
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < 1e-12 or target_norm < 1e-12:
        raise ValueError("Rotation vectors must have nonzero length.")

    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    cross_norm = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if cross_norm < 1e-12:
        if cosine > 0.0:
            return np.eye(3)

        # 반대 방향인 경우 source에 수직인 축을 골라 180도 회전
        trial_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(source, trial_axis)) > 0.9:
            trial_axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial_axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / cross_norm**2)


def main():
    atoms = read(INPUT_POSCAR, format="vasp")
    cell = atoms.cell.array
    positions = atoms.get_positions()
    pbc = atoms.get_pbc()

    nn_vector, _ = find_mic(
        positions[n2_index] - positions[n1_index],
        cell=cell,
        pbc=pbc,
    )
    target_nn = lattice_direction(NN_UVW, cell)
    rotation = rotation_matrix_from_vectors(nn_vector, target_nn)

    # 분율좌표 (0.5, 0.5, 0.5)를 Cartesian 회전 중심으로 변환
    center = ROTATION_CENTER_SCALED @ cell
    for index in fa_indices:
        relative_vector, _ = find_mic(
            positions[index] - center,
            cell=cell,
            pbc=pbc,
        )
        positions[index] = center + rotation @ relative_vector

    atoms.set_positions(positions)
    atoms.wrap()
    write(
        OUTPUT_POSCAR,
        atoms,
        format="vasp",
        direct=True,
        sort=False,
        vasp5=True,
    )

    new_nn, _ = find_mic(
        atoms.positions[n2_index] - atoms.positions[n1_index],
        cell=atoms.cell.array,
        pbc=atoms.get_pbc(),
    )
    cosine = np.dot(new_nn, target_nn) / np.linalg.norm(new_nn)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    direction_label = " ".join(str(component) for component in NN_UVW)

    print(f"Saved: {OUTPUT_POSCAR}")
    print(
        f"N1 -> N2와 +[{direction_label}] 사이의 각도: "
        f"{angle:.8f} degrees"
    )


if __name__ == "__main__":
    main()
