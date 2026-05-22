import os
import shutil
from typing import Sequence

import numpy as np
import ase.io
from ase import Atoms
from ase.mep import AutoNEB
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator

from qc_launcher.drivers.base_driver import BaseDriver


def run_autoneb(
    driver: BaseDriver,
    config: dict,
    atoms_list: Sequence[Atoms],
    filename: str = "molecule",
) -> None:

    # check if all images have the same atoms
    symbols_list = [atoms.symbols for atoms in atoms_list]
    if not all(np.array_equal(symbols, symbols_list[0]) for symbols in symbols_list):
        raise ValueError("All images must have the same atoms (same symbols in the same order).")

    # get config
    max_images = config.get("max_images", 11)
    fmax = float(config.get("fmax", 0.05))
    climb = config.get("climb", False)
    max_opt_steps = config.get("max_opt_steps", None)
    max_neb_steps = config.get("max_neb_steps", 200)
    spring_constant = float(config.get("spring_constant", 0.1))
    neb_method = config.get("neb_method", "improvedtangent")
    space_energy_ratio = float(config.get("space_energy_ratio", 0.5))
    smooth_curve = config.get("smooth_curve", False)
    fixed_reactant: bool = config.get("fixed_reactant", False)
    fixed_product: bool = config.get("fixed_product", False)
    interpolate_method: str = config.get("interpolate_method", "idpp")
    clear_scratch = config.get("clear_scratch", False)

    # read the input atoms
    if len(atoms_list) != 2:
        raise ValueError("Input must contain exactly two images (initial and final geometries).")
    init_atoms, final_atoms = atoms_list[0], atoms_list[1]

    # optimize terminal images first
    init_atoms.calc = driver.to_ase_calc()
    os.makedirs("scratch", exist_ok=True)
    if not fixed_reactant:
        print("Optimizing initial image...")
        with FIRE(init_atoms, maxstep=max_opt_steps) as opt:
            opt.run(fmax=fmax)
    sp_calc = SinglePointCalculator(
        init_atoms, energy=init_atoms.get_potential_energy(), forces=init_atoms.get_forces()
    )
    init_atoms.calc = sp_calc
    ase.io.write(f"scratch/{filename}000.traj", init_atoms)
    final_atoms.calc = driver.to_ase_calc()
    if not fixed_product:
        print("Optimizing final image...")
        with FIRE(final_atoms, maxstep=max_opt_steps) as opt:
            opt.run(fmax=fmax)
    sp_calc = SinglePointCalculator(
        final_atoms, energy=final_atoms.get_potential_energy(), forces=final_atoms.get_forces()
    )
    final_atoms.calc = sp_calc
    ase.io.write(f"scratch/{filename}001.traj", final_atoms)

    def attach_calculators(images: Sequence[Atoms]) -> None:
        for image in images:
            if driver.atoms is not None:
                image.info["charge"] = driver.atoms.info.get("charge", 0)
                image.info["multiplicity"] = driver.atoms.info.get("multiplicity", 1)
            image_driver = type(driver)(image.copy(), dict(driver.config), verbose=driver.verbose)
            image.calc = image_driver.to_ase_calc()

    autoneb = AutoNEB(
        attach_calculators=attach_calculators,
        prefix=f"scratch/{filename}",
        n_simul=1,
        n_max=max_images,
        iter_folder=os.path.join(os.getcwd(), "scratch"),
        climb=climb,
        fmax=fmax,
        maxsteps=max_neb_steps,
        k=spring_constant,
        method=neb_method,
        optimizer=FIRE,
        space_energy_ratio=space_energy_ratio,
        parallel=False,
        smooth_curve=smooth_curve,
        interpolate_method=interpolate_method,
    )
    autoneb.run()

    images = autoneb.all_images

    # save final images
    ase.io.write(f"{filename}_neb.xyz", images)
    # print energies of all images
    energies = np.array([image.get_potential_energy() for image in images])
    for i, energy in enumerate(energies):
        print(f"Image {i:02d}: Energy = {energy:.6f} eV")

    ts_index = np.argmax(energies)
    print(f"Transition state is image {ts_index:02d} with energy {energies[ts_index]:.6f} eV")
    ase.io.write(f"{filename}_ts.xyz", images[ts_index])

    # clear scratch
    if clear_scratch:
        scratch_dir = os.path.join(os.getcwd(), "scratch")
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir)