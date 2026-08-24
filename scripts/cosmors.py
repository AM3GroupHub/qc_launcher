import argparse
import os
import io

import numpy as np
import ase.io
from scipy.optimize import fsolve
from pyscf import gto, dft
from pyscf.solvent import pcm
from pyscf.solvent.cosmors import write_cosmo_file
from opencosmorspy.parameterization import openCOSMORS24a
from opencosmorspy.cosmors import COSMORS


def compute_ln_gamma(crs: COSMORS, mole_fractions: list[float], temp: float) -> float:
    """Compute ln(gamma) for a given composition at specified temperature using openCOSMO-RS"""
    crs.clear_jobs()
    crs.add_job(np.array(mole_fractions), temp, refst='pure_component')
    return crs.calculate()['tot']['lng'][0][0]


def solubility(crs: COSMORS, delta_h_fus: float, t_fus: float, temp: float = 298.15, iterative: bool = False) -> float:
    """Estimate solubility from openCOSMO-RS ln(gamma)"""
    R = 8.3145  # Gas constant in J/(mol*K)
    rhs = -delta_h_fus / R * (1 / temp - 1 / t_fus)

    if not iterative:
        ln_gamma_inf = compute_ln_gamma(crs, [0.0, 1.0], temp)
        return np.exp(rhs - ln_gamma_inf)
    
    def equation(x_guess: float) -> float:
        """Equilibrium condition to be solved"""
        x_guess = np.clip(x_guess, 1e-15, 1).item()  # Ensure x_guess is within bounds
        x = np.array([x_guess, 1 - x_guess])
        ln_gamma = compute_ln_gamma(crs, x, temp)
        return np.abs(ln_gamma + np.log(x_guess) - rhs)

    result = fsolve(equation, 1e-5)
    
    x_sol = np.clip(result[0], 1e-15, 1)  # Ensure the solution is within bounds
    return x_sol


def main():
    parser = argparse.ArgumentParser(description="Generate COSMO-RS input files from a molecule")
    parser.add_argument("--solute", required=True, help="Path to the XYZ file of the solute molecule")
    parser.add_argument("--solvent", required=True, help="Path to the XYZ file of the solvent molecule")
    parser.add_argument("--basis", default="def2-SVP", help="Basis set for the calculation (default: def2-SVP)")
    parser.add_argument("--xc", default="wB97M-D3BJ", help="DFT functional for the calculation (default: B3LYP)")
    parser.add_argument("--delta-h-fus", type=float, default=10000, help="Enthalpy of fusion in J/mol (default: 10000)")
    parser.add_argument("--t-fus", type=float, default=350, help="Fusion temperature in K (default: 350)")
    parser.add_argument("--temp", type=float, default=298.15, help="Temperature in K (default: 298.15)")
    args = parser.parse_args()

    # read input file and create a molecule object
    cosmo_files = []
    for inputfile in [args.solute, args.solvent]:
        filename = os.path.basename(os.path.splitext(inputfile)[0])
        if not os.path.exists(inputfile):
            raise FileNotFoundError(f"Input file '{inputfile}' not found.")
        cosmo_file = f"{filename}.cosmo"
        cosmo_files.append(cosmo_file)
        if os.path.exists(cosmo_file):
            print(f"COSMO file '{cosmo_file}' already exists. Skipping generation.")
            continue

        atoms = ase.io.read(inputfile)
        charge = atoms.info.get("charge", 0)
        multiplicity = atoms.info.get("multiplicity", 1)
        mol = gto.M(
            atom=[(atom.symbol, atom.position) for atom in atoms],
            charge=charge,
            spin=multiplicity - 1,
            basis=args.basis,
        )
        # configure PCM
        cm = pcm.PCM(mol)
        cm.eps = float("inf")  # f_epsilon = 1 is required for COSMO-RS
        cm.method = "IEF-PCM"
        cm.lebedev_order = 29

        # Perform DFT calculation with PCM solvent model
        mf = dft.KS(mol, xc=args.xc)
        mf = mf.PCM(cm)
        mf.kernel()

        # generate COSMO-file
        with open(cosmo_file, "w") as f:
            write_cosmo_file(f, mf)

    # set up openCOSMO-RS
    crs = COSMORS(par=openCOSMORS24a())
    solute_file = cosmo_files[0]
    solvent_file = cosmo_files[1]

    print("Solubility results in mole fractions:")
    crs.clear_molecules()
    crs.add_molecule([solute_file])
    crs.add_molecule([solvent_file])
    x_non_iater = solubility(crs, delta_h_fus=args.delta_h_fus, t_fus=args.t_fus, temp=args.temp, iterative=False)
    print(f"Non-iterative solubility: {x_non_iater:.6f}")
    x_iter = solubility(crs, delta_h_fus=args.delta_h_fus, t_fus=args.t_fus, temp=args.temp, iterative=True)
    print(f"Iterative solubility: {x_iter:.6f}")


if __name__ == "__main__":
    main()