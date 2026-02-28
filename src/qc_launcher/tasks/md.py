import time

import numpy as np
from ase.units import fs, Pascal
from ase.optimize import FIRE, FIRE2, LBFGS
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT, IsotropicMTKNPT, MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.logger import MDLogger
from ase.constraints import FixAtoms

from qc_launcher.drivers import BaseDriver


def run_md(
    driver: BaseDriver,
    config: dict,
    filename: str = "molecule",
) -> None:
    """
    Run molecular dynamics simulation using various ensembles (NVE, NVT, NPT).
    
    Supports different integrators:
    - VelocityVerlet: NVE ensemble (constant energy)
    - Langevin: NVT ensemble (constant temperature with friction)
    - NPT: NPT ensemble (constant temperature and pressure)
    """
    # record the start time
    start_time = time.time()

    # get atoms object from driver
    atoms = driver.atoms

    # get MD configuration
    ensemble = config.get("ensemble", "NVT")
    minimize = config.get("minimize", True)
    optimizer = config.get("optimizer", "FIRE")
    opt_kwargs = config.get("opt_kwargs", {})
    fmax = float(config.get("fmax", 0.05))  # eV/Å
    max_min_steps = int(config.get("max_min_steps", 500))
    min_trajectory = config.get("min_trajectory", f"{filename}_min.traj")
    timestep = float(config.get("timestep", 1.0))  # fs
    temperature = float(config.get("temperature", 300.0))  # Kelvin
    nsteps = int(config.get("nsteps", 1000))
    loginterval = int(config.get("loginterval", 100))
    trajectory = config.get("trajectory", f"{filename}_md.traj")
    logfile = config.get("logfile", "-")  # "-" means print to stdout
    append_trajectory = config.get("append_trajectory", False)
    init_velocities = config.get("init_velocities", True)
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed=seed)  # For reproducibility

    # read atoms tags to get constraints
    fix_mask = np.asarray(atoms.get_tags(), dtype=bool)
    fix_indices = np.where(fix_mask)[0]
    if np.any(fix_indices):
        constraints = FixAtoms(indices=fix_indices)
        atoms.set_constraint(constraints)
    
    # set calculator
    atoms.calc = driver.to_ase_calc()
    
    # pre-minimize the structure before starting MD
    if minimize:
        if optimizer == "FIRE":
            opt = FIRE(atoms, logfile=logfile, trajectory=min_trajectory, **opt_kwargs)
        elif optimizer == "FIRE2":
            opt = FIRE2(atoms, logfile=logfile, trajectory=min_trajectory, **opt_kwargs)
        elif optimizer == "LBFGS":
            opt = LBFGS(atoms, logfile=logfile, trajectory=min_trajectory, **opt_kwargs)
        else:
            raise ValueError(f"Unknown optimization engine: {optimizer}.")
        print(f"Pre-minimizing using {optimizer} ...")
        opt.run(fmax=fmax, steps=max_min_steps)

    # initialize velocities
    if init_velocities:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=rng)
        Stationary(atoms)
        ZeroRotation(atoms)
    
    # setup MD integrator based on ensemble
    if ensemble in {"NVE", "VelocityVerlet"}:
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep * fs,
            trajectory=trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory
        )
    elif ensemble == "Langevin":
        friction = float(config.get("friction", 0.002))  # fs^-1
        dyn = Langevin(
            atoms,
            timestep=timestep * fs,
            temperature_K=temperature,
            friction=friction / fs,
            trajectory=trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
    elif ensemble in {"NVT", "NoseHooverChainNVT"}:
        tdamp = float(config.get("dtamp", timestep * 100))  # fs
        tchain = int(config.get("tchain", 3))
        tloop = int(config.get("tloop", 1))
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep * fs,
            temperature_K=temperature,
            tdamp=tdamp * fs,
            tchain=tchain,
            tloop=tloop,
            trajectory=trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
    elif ensemble in {"NPT", "MTKNPT"}:
        pressure = float(config.get("pressure", 101325))  # Pa
        tdamp = float(config.get("dtamp", timestep * 100))  # fs
        pdamp = float(config.get("pdamp", timestep * 1000))  # fs
        tchain = int(config.get("tchain", 3))
        tloop = int(config.get("tloop", 1))
        pchain = int(config.get("pchain", 3))
        ploop = int(config.get("ploop", 1))
        dyn = MTKNPT(
            atoms,
            timestep=timestep * fs,
            temperature_K=temperature,
            pressure_au=pressure * Pascal,
            tdamp=tdamp * fs,
            pdamp=pdamp * fs,
            tchain=tchain,
            pchain=pchain,
            tloop=tloop,
            ploop=ploop,
            trajectory=trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
    elif ensemble == "IsotropicMTKNPT":
        pressure = float(config.get("pressure", 101325))  # Pa
        tdamp = float(config.get("dtamp", timestep * 100))  # fs
        pdamp = float(config.get("pdamp", timestep * 1000))  # fs
        tchain = int(config.get("tchain", 3))
        tloop = int(config.get("tloop", 1))
        pchain = int(config.get("pchain", 3))
        ploop = int(config.get("ploop", 1))
        dyn = IsotropicMTKNPT(
            atoms,
            timestep=timestep * fs,
            temperature_K=temperature,
            pressure_au=pressure * Pascal,
            tdamp=tdamp * fs,
            pdamp=pdamp * fs,
            tchain=tchain,
            pchain=pchain,
            tloop=tloop,
            ploop=ploop,
            trajectory=trajectory,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
    else:
        raise ValueError(f"Unknown ensemble: {ensemble}.")

    md_logger = MDLogger(dyn=dyn, atoms=atoms, logfile=logfile, stress=True)
    dyn.attach(md_logger, interval=loginterval)
    
    # run MD simulation
    print(f"Starting MD simulation for {nsteps} steps...")
    print(f"Timestep: {timestep} fs")
    print(f"Total simulation time: {nsteps * timestep / 1000:.2f} ps")
    
    dyn.run(nsteps)
    
    # record end time
    end_time = time.time()
    print(f"MD simulation completed in {end_time - start_time:.2f} seconds.")
