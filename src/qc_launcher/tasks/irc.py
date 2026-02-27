import time

import ase.io
from ase.units import Hartree, Bohr
from sella import IRC

from qc_launcher.drivers import BaseDriver


def run_irc(
    driver: BaseDriver,
    config: dict,
    filename: str = "molecule",
) -> None:
    # record the start time
    start_time = time.time()
    
    atoms = driver.atoms

    # get config
    irc_trajectory = config.get("irc_trajectory", f"{filename}_irc.traj")
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
        hessian_func=hessian_func,
    )

    # forward direction
    pos_init = atoms.get_positions().copy()
    if direction in {"forward", "both"}:
        print("Starting forward IRC...")
        converged = irc.run(fmax=fmax_criterion, steps=steps, direction="forward")
        if not converged:
            Warning("Forward IRC did not converge within the maximum number of steps.")
        ase.io.write(f"{filename}_forward.xyz", irc.atoms, columns=["symbols", "positions"])

    # reverse direction
    if direction in {"reverse", "both"}:
        print("Starting reverse IRC...")
        # reset positions to initial geometry
        irc.v0ts = None
        atoms.set_positions(pos_init)
        converged = irc.run(fmax=fmax_criterion, steps=steps, direction="reverse")
        if not converged:
            Warning("Reverse IRC did not converge within the maximum number of steps.")
        ase.io.write(f"{filename}_reverse.xyz", irc.atoms, columns=["symbols", "positions"])
    
    end_time = time.time()
    print(f"IRC completed in {end_time - start_time:.2f} seconds.")
