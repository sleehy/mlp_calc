path = "plot_archive/mace_mp/thermal_conductivity/lbte/inputs/kappa-m202020.hdf5"
from pathlib import Path

import h5py
import numpy as np


def print_hdf5_structure(
    file_path: str | Path,
    *,
    preview: bool = True,
    max_preview_items: int = 10,
) -> None:
    """
    HDF5 파일의 전체 그룹 및 데이터셋 구조를 출력합니다.

    Parameters
    ----------
    file_path
        확인할 HDF5 파일 경로
    preview
        각 데이터셋의 일부 값을 출력할지 여부
    max_preview_items
        미리 출력할 최대 원소 개수
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 파일을 찾을 수 없습니다: {file_path}")

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        depth = name.count("/")
        indent = "    " * depth
        short_name = name.split("/")[-1]

        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {short_name}/")

            if obj.attrs:
                for key, value in obj.attrs.items():
                    print(f"{indent}    Attribute: {key} = {value}")

        elif isinstance(obj, h5py.Dataset):
            print(
                f"{indent}[Dataset] {short_name} "
                f"shape={obj.shape}, dtype={obj.dtype}"
            )

            if obj.attrs:
                for key, value in obj.attrs.items():
                    print(f"{indent}    Attribute: {key} = {value}")

            if preview:
                try:
                    data = obj[()]

                    if np.isscalar(data):
                        print(f"{indent}    Value: {data}")
                    else:
                        flattened = np.asarray(data).reshape(-1)
                        sample = flattened[:max_preview_items]
                        print(f"{indent}    Preview: {sample}")

                        if flattened.size > max_preview_items:
                            print(
                                f"{indent}    "
                                f"... total {flattened.size} elements"
                            )

                except Exception as error:
                    print(f"{indent}    Preview failed: {error}")

    print(f"File: {file_path.resolve()}")

    with h5py.File(file_path, "r") as hdf5_file:
        if hdf5_file.attrs:
            print("\n[File attributes]")
            for key, value in hdf5_file.attrs.items():
                print(f"  {key} = {value}")

        print("\n[HDF5 structure]")
        hdf5_file.visititems(visitor)


if __name__ == "__main__":
    print_hdf5_structure(
        path,
        preview=True,
        max_preview_items=10,
    )