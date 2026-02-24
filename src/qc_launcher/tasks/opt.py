import time
import logging
from typing import Dict

from ase import Atoms
from ase.units import Hartree, Bohr
from sella import Sella, Constraints

from qc_launcher.drivers import BaseQCCalculator


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


def run_optimization_task(
    atoms: Atoms,
    config: Dict,
    task_name: str = "mytask",
):
    # record the start time
    start_time = time.time()
    # get configs
    optts = config.get("ts", False)
    if optts:
        eig = config.get("calc_hess", True)
        order = 1
    else:
        eig = config.get("calc_hess", False)
        order = 0
    # constraints
    cons_dict = config.get("constraints", {})
    cons = setup_constraints(atoms, cons_dict)

    calc: BaseQCCalculator = atoms.calc
    # set sella optimizer
    sella = Sella(
        atoms=atoms,
        trajectory=config.get("trajectory", f"{task_name}_opt.traj"),
        order=order,
        internal=config.get("internal", True),
        constraints=cons,
        constraints_tol=config.get("constraints_tol", 1e-5),
        delta0=float(config.get("delta0", 0.1)),
        eta=float(config.get("eta", 1e-4)),
        gamma=float(config.get("gamma", 0.1)),
        eig=eig,
        threepoint=True,
        nsteps_per_diag=config.get("nsteps_per_diag", 3),
        diag_every_n=config.get("diag_every_n", None),
        hessian_function=calc.compute_hessian,
    )
    energy_criterion = float(config.get("", 1e-6)) * Hartree
    force_criterion = float(config.get("fmax", 1e-3)) * Hartree / Bohr