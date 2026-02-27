import time

import numpy as np
import ase.io
from ase import Atoms
from ase.units import Hartree, Bohr
from sella import Sella, Constraints

from qc_launcher.drivers import BaseDriver


def setup_constraints(atoms: Atoms, cons_dict: dict) -> Constraints:
    """
    Setup constraints for the atoms object.
    
    Args:
        atoms: ASE Atoms object
        cons_dict: Dictionary containing constraint specifications with keys:
                   - "fix_translation": list of atom indices
                   - "fix_bond": list of [atom_i, atom_j] pairs
                   - "fix_angle": list of [atom_i, atom_j, atom_k] triplets
                   - "fix_dihedral": list of [atom_i, atom_j, atom_k, atom_l] quartets
    
    Returns:
        Constraints object or None if cons_dict is empty
    """
    if not cons_dict:
        return None
    
    cons = Constraints(atoms)
    
    # Define constraint types and their corresponding methods
    constraint_specs = {
        "fix_translation": {
            "method": "fix_translation",
            "args_processor": lambda x: (x,),
            "label": "translation"
        },
        "fix_bond": {
            "method": "fix_bond",
            "args_processor": lambda x: (tuple(x),),
            "label": "bond"
        },
        "fix_angle": {
            "method": "fix_angle",
            "args_processor": lambda x: (tuple(x),),
            "label": "angle"
        },
        "fix_dihedral": {
            "method": "fix_dihedral",
            "args_processor": lambda x: (tuple(x),),
            "label": "dihedral"
        }
    }
    
    # Apply constraints
    for constraint_type, spec in constraint_specs.items():
        if constraint_type in cons_dict:
            constraint_items = cons_dict[constraint_type]
            for item in constraint_items:
                args = spec["args_processor"](item)
                getattr(cons, spec["method"])(*args)
            print(f"Applied {spec['label']} constraints: {constraint_items}")
    
    return cons


def run_opt(
    driver: BaseDriver,
    config: dict,
    input_name: str = "molecule",
) -> BaseDriver:
    """
    Run geometry optimization using the Sella optimizer with the specified driver and configuration.
    """
    # record the start time
    start_time = time.time()

    # get atoms object from driver
    atoms = driver.atoms

    # get configs
    optts = config.get("ts", False)
    if optts:
        eig = config.get("calc_hess", True)
        order = 1
    else:
        eig = config.get("calc_hess", False)
        order = 0
    tractory = config.get("trajectory", f"{input_name}_opt.traj")
    internal = config.get("internal", True)
    delta0 = float(config.get("delta0", 0.1))
    eta = float(config.get("eta", 1e-4))
    gamma = float(config.get("gamma", 0.1))
    nsteps_per_diag = config.get("nsteps_per_diag", 3)  # calculate exact hessian every n steps, when approximate hessian has inaccurate number of imaginary frequencies
    diag_every_n = config.get("diag_every_n", None)  # force calculation of exact hessian every n steps, regardless of the number of imaginary frequencies
    # constraints
    cons_dict = config.get("constraints", {})
    cons = setup_constraints(atoms, cons_dict)
    constraints_tol = config.get("constraints_tol", 1e-5)
    # convergence criteria
    ediff_criterion = float(config.get("ediff", 1e-6)) * Hartree
    fmax_criterion = float(config.get("fmax", 4.5e-4)) * Hartree / Bohr
    frms_criterion = float(config.get("frms", 3.0e-4)) * Hartree / Bohr
    dmax_criterion = float(config.get("dmax", 1.8e-3))
    drmx_criterion = float(config.get("drmx", 1.2e-3))
    max_steps = config.get("max_steps", 150)

    # set sella optimizer
    atoms.calc = driver.to_ase_calc()
    hessian_func = lambda x: driver.compute_hessian(x, use_cache=True, hess_format="ase")
    sella = Sella(
        atoms=atoms,
        trajectory=tractory,
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
        hessian_function=hessian_func,
    )
    # run optimization
    last_pos = atoms.get_positions().copy()
    last_energy = np.inf
    for i in sella.irun(fmax=0, steps=max_steps):
        delta_pos = np.linalg.norm(atoms.get_positions() - last_pos, axis=1)
        delta_energy = abs(atoms.get_potential_energy() - last_energy)
        fmax = np.max(np.abs(atoms.get_forces()))
        frms = np.sqrt(np.mean(atoms.get_forces()**2))
        dmax = np.max(delta_pos)
        drms = np.sqrt(np.mean(delta_pos**2))
        if (delta_energy < ediff_criterion and
            fmax < fmax_criterion and
            frms < frms_criterion and
            dmax < dmax_criterion and
            drms < drmx_criterion):
            print("Optimization converged!")
            break
        last_pos = atoms.get_positions().copy()
        last_energy = atoms.get_potential_energy()
    else:
        print("Optimization did not converge within the maximum number of steps.")
        print(f"Final Energy Change   : {delta_energy:.6e} Eh")
        print(f"Final MAX force       : {fmax * Bohr / Hartree:.6e} Eh/Bohr")
        print(f"Final RMS force       : {frms * Bohr / Hartree:.6e} Eh/Bohr")
        print(f"Final MAX displacement: {dmax:.6e} Angstrom")
        print(f"Final RMS displacement: {drms:.6e} Angstrom")
    
    # save final structure
    opt_outputfile = config.get("outputfile", f"{input_name}_opt.xyz")
    ase.io.write(opt_outputfile, atoms, columns=["symbols", "positions"])
    
    # record end time
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

    driver.update_atoms(atoms)
    return driver
