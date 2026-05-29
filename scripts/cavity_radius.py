import argparse
import math

import numpy as np
import ase.io
from ase.units import Bohr
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.solvent import pcm


def main():
    parser = argparse.ArgumentParser(description="Calculate cavity radius for PCM")
    parser.add_argument("xyz_file", type=str, help="Path to the XYZ file of the molecule")
    args = parser.parse_args()

    # Load molecule from XYZ file
    atoms = ase.io.read(args.xyz_file)
    charge = atoms.info.get("charge", 0)
    multiplicity = atoms.info.get("multiplicity", 1)
    mol = gto.M(
        atom=[(atoms[i].symbol, atoms[i].position) for i in range(len(atoms))],
        charge=charge,
        spin=multiplicity - 1,
    )

    # generate PCM surface
    radii_table = 1.2 * pcm.modified_Bondi
    ng = gen_grid.LEBEDEV_ORDER[29]
    surface = pcm.gen_surface(mol, rad=radii_table, ng=ng)
    
    # calculate cavity radius
    coords: np.ndarray = surface["grid_coords"]
    normals: np.ndarray = surface["norm_vec"]
    area: np.ndarray = surface["area"]

    origin = np.average(coords, axis=0, weights=area)
    r = coords - origin

    volume_bohr3 = (np.sum(r * normals, axis=1) @ area / 3.0).item()
    if volume_bohr3 < 0:
        raise ValueError("Calculated volume is negative, check the input molecule and PCM parameters.")
    area_bohr2 = np.sum(area).item()
    radius_bohr = (3 * volume_bohr3 / (4 * math.pi)) ** (1/3)
    radius_from_area_bohr = math.sqrt(area_bohr2 / (4.0 * math.pi))

    print(f"Cavity radius (from volume): {radius_bohr:.4f} Bohr")
    print(f"Cavity radius (from volume): {radius_bohr * Bohr:.4f} Angstrom")
    print(f"Cavity radius (from area): {radius_from_area_bohr:.4f} Bohr")
    print(f"Cavity radius (from area): {radius_from_area_bohr * Bohr:.4f} Angstrom")
    print(f"Average Radius (from volume and area): {((radius_bohr + radius_from_area_bohr) / 2):.4f} Bohr")
    print(f"Average Radius (from volume and area): {((radius_bohr + radius_from_area_bohr) / 2) * Bohr:.4f} Angstrom")


if __name__ == "__main__":
    main()
