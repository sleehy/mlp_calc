import spglib
from ase.io import read

atoms = read("runs/sevennet/orthohrombic/01_relax/INPUT/POSCAR")

cell = (
    atoms.cell.array,
    atoms.get_scaled_positions(),
    atoms.numbers
)

for symprec in [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    print(
        symprec,
        dataset.international,
        dataset.number
    )