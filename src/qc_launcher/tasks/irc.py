import time

import numpy as np
import ase.io
from ase.units import Hartree, Bohr
from sella import IRC

from qc_launcher.drivers import BaseDriver


def run_irc(
    driver: BaseDriver,
    config: dict,
    filename: str = "molecule",
) -> dict:
    # record the start time
    start_time = time.time()
    
    atoms = driver.atoms

    # get config
    irc_trajectory = config.get("irc_trajectory", f"{filename}_irc.traj")
    irc_output = config.get("irc_output", f"{filename}_irc.xyz")
    ninner_iter = config.get("ninner_iter", 10)
    dx = float(config.get("dx", 0.1))  # step size in Angstrom
    eta = float(config.get("eta", 1e-4))
    keep_going = config.get("keep_going", False)
    diag_every_n = config.get("diag_every_n", None)
    fmax_criterion = float(config.get("fmax", 4.5e-4)) * Hartree / Bohr
    steps = config.get("steps", 10)
    direction = config.get("direction", "both")  # "forward", "backward", or "both"
    assert direction in ["forward", "reverse", "both"], "Invalid direction. Must be 'forward', 'backward', or 'both'."

    # set sella IRC
    atoms.calc = driver.to_ase_calc()
    hessian_func = lambda x: driver.compute_hessian(x, use_cache=True, hess_format="ase")
    irc = IRC(
        atoms=atoms,
        trajectory=irc_trajectory,
        ninner_iter=ninner_iter,
        dx=dx,
        eta=eta,
        peskwargs={"threepoint": True},
        keep_going=keep_going,
        diag_every_n=diag_every_n,
        hessian_function=hessian_func,
    )

    # forward direction
    pos_init = atoms.get_positions().copy()
    irc_atoms_traj = []
    traj_offset = 0
    if direction in {"forward", "both"}:
        print("Starting forward IRC...")
        converged = irc.run(fmax=fmax_criterion, steps=steps, direction="forward")
        if not converged:
            Warning("Forward IRC did not converge within the maximum number of steps.")
        forward_traj = ase.io.Trajectory(irc_trajectory, "r")
        irc_atoms_traj.extend([frame for frame in forward_traj])  # skip the first frame which is the initial geometry
        traj_offset = len(irc_atoms_traj)

    # reverse direction
    if direction in {"reverse", "both"}:
        print("Starting reverse IRC...")
        # reset positions to initial geometry
        irc.v0ts = None
        atoms.set_positions(pos_init)
        converged = irc.run(fmax=fmax_criterion, steps=steps, direction="reverse")
        if not converged:
            Warning("Reverse IRC did not converge within the maximum number of steps.")
        backward_traj = ase.io.Trajectory(irc_trajectory, "r")[traj_offset + 1:]  # read the new frames added by reverse IRC
        irc_atoms_traj = [frame for frame in backward_traj[::-1]] + irc_atoms_traj  # reverse direction frames should be added before the forward direction frames

    # save the full IRC trajectory
    ase.io.write(irc_output, irc_atoms_traj)

    end_time = time.time()
    print(f"IRC completed in {end_time - start_time:.2f} seconds.")
    
    traj_pos = np.stack([atoms.get_positions() for atoms in irc_atoms_traj])
    traj_energies = np.array([atoms.get_potential_energy() for atoms in irc_atoms_traj])
    traj_forces = np.stack([atoms.get_forces() for atoms in irc_atoms_traj])
    return {"irc_traj": {
        "positions": traj_pos,
        "energies": traj_energies,
        "forces": traj_forces
    }}