"""Intrinsic reaction coordinate with Sella's IRC, forward and/or reverse from a transition state."""

from __future__ import annotations

import ase.io
from sella import IRC

from qc_launcher.drivers import BaseDriver

from .common import FMAX_DEFAULT, sella_hessian_function, timed

DIRECTIONS = ("forward", "reverse", "both")


def run_irc(driver: BaseDriver, config: dict, filename: str = "molecule") -> None:
    """Config keys (defaults): ``irc_trajectory``, ``irc_output``, ``ninner_iter`` (10), ``dx`` [Angstrom] (0.1),
    ``eta`` (1e-4), ``keep_going`` (False), ``diag_every_n``, ``fmax`` [eV/Angstrom], ``steps`` (10),
    ``direction`` (forward | reverse | both; ``backward`` is accepted for reverse).
    """
    with timed("IRC"):
        atoms = driver.atoms
        irc_trajectory = config.get("irc_trajectory", f"{filename}_irc.traj")
        irc_output = config.get("irc_output", f"{filename}_irc.xyz")
        ninner_iter = int(config.get("ninner_iter", 10))
        dx = float(config.get("dx", 0.1))
        eta = float(config.get("eta", 1e-4))
        keep_going = bool(config.get("keep_going", False))
        diag_every_n = config.get("diag_every_n")
        fmax_criterion = float(config.get("fmax", FMAX_DEFAULT))
        steps = int(config.get("steps", 10))
        direction = str(config.get("direction", "both")).lower()
        if direction == "backward":
            direction = "reverse"
        if direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction '{direction}'. Use one of {DIRECTIONS}.")

        atoms.calc = driver.to_ase_calc()
        irc = IRC(
            atoms=atoms,
            trajectory=irc_trajectory,
            ninner_iter=ninner_iter,
            dx=dx,
            eta=eta,
            peskwargs={"threepoint": True},
            keep_going=keep_going,
            diag_every_n=diag_every_n,
            hessian_function=sella_hessian_function(driver),
        )

        pos_init = atoms.get_positions().copy()
        frames = []
        n_forward = 0
        if direction in ("forward", "both"):
            print("Starting forward IRC...")
            if not irc.run(fmax=fmax_criterion, steps=steps, direction="forward"):
                print("Warning: forward IRC did not converge within the maximum number of steps.")
            frames = list(ase.io.Trajectory(irc_trajectory, "r"))
            n_forward = len(frames)
        if direction in ("reverse", "both"):
            print("Starting reverse IRC...")
            irc.v0ts = None  # let Sella re-derive the transition vector for the other direction
            atoms.set_positions(pos_init)
            if not irc.run(fmax=fmax_criterion, steps=steps, direction="reverse"):
                print("Warning: reverse IRC did not converge within the maximum number of steps.")
            # new frames only; the first one repeats the transition state
            reverse_frames = list(ase.io.Trajectory(irc_trajectory, "r"))[n_forward + 1 :]
            frames = reverse_frames[::-1] + frames  # reactant -> TS -> product

        ase.io.write(irc_output, frames)
