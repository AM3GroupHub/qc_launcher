"""Molecular dynamics with ASE integrators (NVE, Langevin, Nose-Hoover chain NVT, MTK / Melchionna NPT)."""

from __future__ import annotations

import os

import ase.io
import numpy as np
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.logger import MDLogger
from ase.md.melchionna import MelchionnaNPT
from ase.md.nose_hoover_chain import MTKNPT, IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
from ase.optimize import FIRE, FIRE2, LBFGS
from ase.units import GPa, Pascal, fs

from qc_launcher.drivers import BaseDriver

from .common import timed

OPTIMIZERS = {"FIRE": FIRE, "FIRE2": FIRE2, "LBFGS": LBFGS}
MTK_NPT_CLASSES = {"NPT": MTKNPT, "MTKNPT": MTKNPT, "IsotropicMTKNPT": IsotropicMTKNPT}


def run_md(driver: BaseDriver, config: dict, filename: str = "molecule") -> None:
    """Config keys (defaults): ``ensemble`` (NVT; NVE | VelocityVerlet | Langevin | NVT | NoseHooverChainNVT |
    NPT | MTKNPT | IsotropicMTKNPT | MelchionnaNPT), ``minimize`` (True) with ``optimizer`` (FIRE | FIRE2 |
    LBFGS), ``opt_kwargs``, ``fmax`` (0.05), ``max_min_steps`` (500), ``min_trajectory``; ``timestep`` [fs] (1.0),
    ``temperature`` [K] (300), ``nsteps`` (1000), ``loginterval`` (100), ``trajectory``, ``logfile`` ('-'),
    ``append_trajectory`` (False), ``init_velocities`` (True), ``log_stress`` (False), ``seed`` (42),
    ``resume`` (False). Langevin: ``friction`` [1/fs] (0.002). NVT/NPT: ``tdamp`` [fs] (100 * timestep),
    ``tchain`` (3), ``tloop`` (1); MTK NPT: ``pressure`` [Pa] (101325), ``pdamp`` [fs] (1000 * timestep),
    ``pchain`` (3), ``ploop`` (1); Melchionna NPT: ``externalstress`` [Pa], ``mask``, ``ttime`` / ``ptime`` [fs],
    ``bulk_modulus`` [GPa] (100). Atoms with a non-zero tag are kept fixed.
    """
    with timed("MD simulation"):
        atoms = driver.atoms
        ensemble = config.get("ensemble", "NVT")
        minimize = bool(config.get("minimize", True))
        optimizer = config.get("optimizer", "FIRE")
        opt_kwargs = config.get("opt_kwargs") or {}
        fmax = float(config.get("fmax", 0.05))
        max_min_steps = int(config.get("max_min_steps", 500))
        min_trajectory = config.get("min_trajectory", f"{filename}_min.traj")
        timestep = float(config.get("timestep", 1.0))
        temperature = float(config.get("temperature", 300.0))
        nsteps = int(config.get("nsteps", 1000))
        loginterval = int(config.get("loginterval", 100))
        trajectory = config.get("trajectory", f"{filename}_md.traj")
        logfile = config.get("logfile", "-")
        append_trajectory = bool(config.get("append_trajectory", False))
        init_velocities = bool(config.get("init_velocities", True))
        log_stress = bool(config.get("log_stress", False))
        rng = np.random.default_rng(seed=config.get("seed", 42))

        existing_steps = 0
        if config.get("resume", False) and os.path.exists(trajectory):
            print(f"Resuming from existing trajectory: {trajectory}")
            frames = ase.io.Trajectory(trajectory)
            atoms = frames[-1]  # the driver keeps its own structure; MD continues on the last frame
            existing_steps = (len(frames) - 1) * loginterval  # the first frame is step 0
            if existing_steps >= nsteps:
                print(f"Existing trajectory already has {existing_steps} steps (>= {nsteps} requested). Nothing to do.")
                return
            nsteps -= existing_steps
            append_trajectory = True
            init_velocities = False
            minimize = False
            print(f"Resuming MD simulation for the remaining {nsteps} steps...")

        fix_indices = np.flatnonzero(atoms.get_tags())
        if fix_indices.size:
            atoms.set_constraint(FixAtoms(indices=fix_indices))

        atoms.calc = driver.to_ase_calc()

        if minimize:
            if optimizer not in OPTIMIZERS:
                raise ValueError(f"Unknown optimizer '{optimizer}'. Use one of {list(OPTIMIZERS)}.")
            print(f"Pre-minimizing using {optimizer} ...")
            opt = OPTIMIZERS[optimizer](atoms, logfile=logfile, trajectory=min_trajectory, **opt_kwargs)
            opt.run(fmax=fmax, steps=max_min_steps)

        if init_velocities:
            MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
            Stationary(atoms)
            ZeroRotation(atoms)

        common = {
            "timestep": timestep * fs,
            "trajectory": trajectory,
            "loginterval": loginterval,
            "append_trajectory": append_trajectory,
        }
        if ensemble in ("NVE", "VelocityVerlet"):
            dyn = VelocityVerlet(atoms, **common)
        elif ensemble == "Langevin":
            friction = float(config.get("friction", 0.002))  # 1/fs
            dyn = Langevin(atoms, temperature_K=temperature, friction=friction / fs, **common)
        elif ensemble in ("NVT", "NoseHooverChainNVT"):
            dyn = NoseHooverChainNVT(
                atoms,
                temperature_K=temperature,
                tdamp=float(config.get("tdamp", timestep * 100)) * fs,
                tchain=int(config.get("tchain", 3)),
                tloop=int(config.get("tloop", 1)),
                **common,
            )
        elif ensemble in MTK_NPT_CLASSES:
            dyn = MTK_NPT_CLASSES[ensemble](
                atoms,
                temperature_K=temperature,
                pressure_au=float(config.get("pressure", 101325)) * Pascal,
                tdamp=float(config.get("tdamp", timestep * 100)) * fs,
                pdamp=float(config.get("pdamp", timestep * 1000)) * fs,
                tchain=int(config.get("tchain", 3)),
                pchain=int(config.get("pchain", 3)),
                tloop=int(config.get("tloop", 1)),
                ploop=int(config.get("ploop", 1)),
                **common,
            )
        elif ensemble == "MelchionnaNPT":
            externalstress = config.get("externalstress", 101325)  # Pa
            if isinstance(externalstress, list):
                externalstress = np.asarray(externalstress, dtype=float)
            ptime = float(config.get("ptime", timestep * 1000))  # fs
            bulk_modulus = float(config.get("bulk_modulus", 100))  # GPa
            dyn = MelchionnaNPT(
                atoms,
                temperature_K=temperature,
                externalstress=externalstress * Pascal,
                mask=np.asarray(config.get("mask", [1, 1, 1]), dtype=bool),
                ttime=float(config.get("ttime", timestep * 100)) * fs,
                pfactor=(ptime * fs) ** 2 * bulk_modulus * GPa,
                **common,
            )
        else:
            raise ValueError(f"Unknown ensemble '{ensemble}'.")

        print(f"Starting MD simulation for {nsteps} steps...")
        print(f"Timestep: {timestep} fs")
        print(f"Total simulation time: {nsteps * timestep / 1000:.2f} ps")
        dyn.attach(MDLogger(dyn=dyn, atoms=atoms, logfile=logfile, stress=log_stress), interval=loginterval)
        # continue the step counter on resume: ASE then skips re-writing the last frame and keeps the time axis
        dyn.nsteps = existing_steps
        dyn.run(nsteps)
