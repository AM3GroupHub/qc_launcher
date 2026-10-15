"""Minimum and transition-state optimization with Sella, Gaussian-style convergence criteria."""

from __future__ import annotations

import ase.io
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from sella import Constraints, Sella

from qc_launcher.drivers import BaseDriver

from .common import FMAX_DEFAULT, sella_hessian_function, timed

CONSTRAINT_TYPES = ("fix_translation", "fix_bond", "fix_angle", "fix_dihedral")


def setup_constraints(atoms: Atoms, cons_dict: dict | None) -> Constraints | None:
    """Build Sella constraints from ``{fix_translation: [i, ...], fix_bond: [[i, j], ...], fix_angle: [[i, j, k],
    ...], fix_dihedral: [[i, j, k, l], ...]}``; returns None when nothing is constrained."""
    if not cons_dict:
        return None
    unknown = set(cons_dict) - set(CONSTRAINT_TYPES)
    if unknown:
        raise ValueError(f"Unknown constraint types {sorted(unknown)}. Use {CONSTRAINT_TYPES}.")
    cons = Constraints(atoms)
    for name in CONSTRAINT_TYPES:
        items = cons_dict.get(name)
        if not items:
            continue
        for item in items:
            getattr(cons, name)(item if name == "fix_translation" else tuple(item))
        print(f"Applied {name} constraints: {items}")
    return cons


def run_opt(driver: BaseDriver, config: dict, filename: str = "molecule") -> None:
    """Config keys (defaults): ``ts`` (False), ``calc_hess`` (= ts), ``internal`` (True), ``trajectory``,
    ``outputfile``, ``delta0`` (0.1), ``eta`` (1e-4), ``gamma`` (0.1), ``nsteps_per_diag`` (3), ``diag_every_n``,
    ``constraints``, ``constraints_tol`` (1e-5), ``max_steps`` (150) and the convergence criteria ``ediff`` [eV],
    ``fmax`` / ``frms`` [eV/Angstrom], ``dmax`` / ``drms`` [Angstrom] (Gaussian defaults).
    """
    with timed("Optimization"):
        atoms = driver.atoms
        optts = bool(config.get("ts", False))
        order = 1 if optts else 0
        eig = bool(config.get("calc_hess", optts))
        trajectory = config.get("trajectory", f"{filename}_opt.traj")
        opt_outputfile = config.get("outputfile", f"{filename}_opt.xyz")
        internal = config.get("internal", True)
        delta0 = float(config.get("delta0", 0.1))
        eta = float(config.get("eta", 1e-4))
        gamma = float(config.get("gamma", 0.1))
        # exact Hessian every n steps when the approximate one has the wrong number of negative eigenvalues ...
        nsteps_per_diag = config.get("nsteps_per_diag", 3)
        # ... or unconditionally every n steps
        diag_every_n = config.get("diag_every_n")
        cons = setup_constraints(atoms, config.get("constraints"))
        constraints_tol = float(config.get("constraints_tol", 1e-5))
        ediff_criterion = float(config.get("ediff", 1e-6 * Hartree))
        fmax_criterion = float(config.get("fmax", FMAX_DEFAULT))
        frms_criterion = float(config.get("frms", 3.0e-4 * Hartree / Bohr))
        dmax_criterion = float(config.get("dmax", 1.8e-3 * Bohr))
        drms_criterion = float(config.get("drms", 1.2e-3 * Bohr))
        cmax_criterion = np.inf if cons is None else constraints_tol
        max_steps = int(config.get("max_steps", 150))

        atoms.calc = driver.to_ase_calc()
        sella = Sella(
            atoms=atoms,
            trajectory=trajectory,
            order=order,
            internal=internal,
            constraints=cons,
            constraints_tol=constraints_tol,
            delta0=delta0,
            eta=eta,
            gamma=gamma,
            eig=eig,
            threepoint=True,
            nsteps_per_diag=nsteps_per_diag,
            diag_every_n=diag_every_n,
            hessian_function=sella_hessian_function(driver),
        )

        last_pos = atoms.get_positions().copy()
        last_energy = np.inf
        delta_energy = fmax = frms = dmax = drms = cmax = np.nan
        converged = False
        for _ in sella.irun(fmax=0.0, steps=max_steps):
            delta_pos = atoms.get_positions() - last_pos
            delta_energy = abs(atoms.get_potential_energy() - last_energy)
            forces = sella.pes.get_projected_forces()
            fmax = np.max(np.linalg.norm(forces, axis=1))
            frms = np.sqrt(np.mean(forces**2))  # RMS over all 3N components, as in Gaussian
            dmax = np.max(np.linalg.norm(delta_pos, axis=1))
            drms = np.sqrt(np.mean(delta_pos**2))
            cmax = np.linalg.norm(sella.pes.get_res())
            if (
                delta_energy < ediff_criterion
                and fmax < fmax_criterion
                and frms < frms_criterion
                and dmax < dmax_criterion
                and drms < drms_criterion
                and cmax < cmax_criterion
            ):
                converged = True
                print("Optimization converged!")
                break
            last_pos = atoms.get_positions().copy()
            last_energy = atoms.get_potential_energy()
        if not converged:
            print("Optimization did not converge within the maximum number of steps.")
            print(f"Final Energy Change   : {delta_energy:.6e} eV, criterion: {ediff_criterion:.6e} eV")
            print(f"Final MAX force       : {fmax:.6e} eV/Å , criterion: {fmax_criterion:.6e} eV/Å")
            print(f"Final RMS force       : {frms:.6e} eV/Å , criterion: {frms_criterion:.6e} eV/Å")
            print(f"Final MAX displacement: {dmax:.6e} Å, criterion: {dmax_criterion:.6e} Å")
            print(f"Final RMS displacement: {drms:.6e} Å, criterion: {drms_criterion:.6e} Å")
            print(f"Final MAX constraint  : {cmax:.6e}, criterion: {cmax_criterion:.6e}")

        ase.io.write(opt_outputfile, atoms, columns=["symbols", "positions"])
