from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read, write

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_POSCAR = SCRIPT_DIR.parent / "t-FAPb" / "POSCAR"
CH_UVW = np.array([1, 0, 0])
UVW_TAG = "".join(str(component) for component in CH_UVW)
OUTPUT_POSCAR = SCRIPT_DIR.parent / "t-FAPb" / f"POSCAR_FA_{UVW_TAG}"
ROTATION_CENTER_SCALED = np.array([0.5, 0.5, 0.5])

# POSCAR에서 해당 FA 분자를 구성하는 원자 번호
# Python 번호이므로 0부터 시작합니다.
fa_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # 반드시 수정

# FA 분자의 중심 C와 C에 직접 붙은 H의 원자 번호
c_index = 0  # 반드시 수정
h_index = 3  # 반드시 수정


def lattice_direction(uvw, cell):
    """결정 방향 [uvw]를 정규화된 Cartesian 벡터로 변환합니다."""
    vector = np.asarray(uvw, dtype=float) @ cell
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("CH_UVW must not be [0, 0, 0].")
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

    # 현재 C -> H 벡터
    ch_vector, _ = find_mic(
        positions[h_index] - positions[c_index],
        cell=cell,
        pbc=pbc,
    )

    target_ch = lattice_direction(CH_UVW, cell)
    rotation = rotation_matrix_from_vectors(ch_vector, target_ch)

    # 분율좌표로 지정한 셀 중심을 Cartesian 회전 중심으로 사용
    center = ROTATION_CENTER_SCALED @ cell

    for i in fa_indices:
        relative_vector, _ = find_mic(
            positions[i] - center,
            cell=cell,
            pbc=pbc,
        )
        positions[i] = center + rotation @ relative_vector

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

    # 결과 확인
    new_ch, _ = find_mic(
        atoms.positions[h_index] - atoms.positions[c_index],
        cell=atoms.cell.array,
        pbc=atoms.get_pbc(),
    )
    ch_cosine = np.dot(new_ch, target_ch) / np.linalg.norm(new_ch)
    ch_angle = np.degrees(np.arccos(np.clip(ch_cosine, -1.0, 1.0)))
    ch_label = " ".join(str(component) for component in CH_UVW)

    print(f"Saved: {OUTPUT_POSCAR}")
    print(f"C -> H와 +[{ch_label}] 사이의 각도: {ch_angle:.8f} degrees")


if __name__ == "__main__":
    main()
