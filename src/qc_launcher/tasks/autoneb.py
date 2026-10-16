"""ASE AutoNEB: grows the band adaptively between two relaxed endpoints."""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence

import ase.io
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.mep import AutoNEB
from ase.optimize import FIRE

from qc_launcher.drivers import BaseDriver

from .common import check_same_species, relax_endpoints, write_path_results


def run_autoneb(driver: BaseDriver, config: dict, atoms_list: Sequence[Atoms], filename: str = "molecule") -> None:
    """Config keys (defaults): ``max_images`` (11), ``fmax`` (0.05), ``climb`` (False), ``max_steps`` (200),
    ``endpoint_max_steps`` (endpoint relaxation step limit), ``spring_constant`` (0.1), ``neb_method``
    (improvedtangent), ``space_energy_ratio`` (0.5), ``smooth_curve`` (False), ``fixed_reactant`` /
    ``fixed_product`` (False), ``interpolate_method`` (idpp), ``scratch_dir`` (scratch), ``clear_scratch`` (False).
    Requires exactly two input frames.
    """
    check_same_species(atoms_list)
    if len(atoms_list) != 2:
        raise ValueError("Input must contain exactly two images (initial and final geometries).")
    max_images = int(config.get("max_images", 11))
    fmax = float(config.get("fmax", 0.05))
    climb = bool(config.get("climb", False))
    max_opt_steps = config.get("endpoint_max_steps")
    max_neb_steps = int(config.get("max_steps", 200))
    spring_constant = float(config.get("spring_constant", 0.1))
    neb_method = config.get("neb_method", "improvedtangent")
    space_energy_ratio = float(config.get("space_energy_ratio", 0.5))
    smooth_curve = bool(config.get("smooth_curve", False))
    fixed_reactant = bool(config.get("fixed_reactant", False))
    fixed_product = bool(config.get("fixed_product", False))
    interpolate_method = config.get("interpolate_method", "idpp")
    scratch_dir = os.path.abspath(config.get("scratch_dir", "scratch"))
    clear_scratch = bool(config.get("clear_scratch", False))

    images = [atoms_list[0], atoms_list[1]]
    os.makedirs(scratch_dir, exist_ok=True)
    relax_endpoints(
        images,
        driver.to_ase_calc(),
        fmax=fmax,
        steps=max_opt_steps,
        fixed_reactant=fixed_reactant,
        fixed_product=fixed_product,
    )
    for index, image in enumerate(images):
        image.calc = driver.to_ase_calc()
        image.calc = SinglePointCalculator(image, energy=image.get_potential_energy(), forces=image.get_forces())
        ase.io.write(os.path.join(scratch_dir, f"{filename}{index:03d}.traj"), image)

    def attach_calculators(new_images: Sequence[Atoms]) -> None:
        for image in new_images:
            if driver.atoms is not None:
                image.info["charge"] = driver.atoms.info.get("charge", 0)
                image.info["multiplicity"] = driver.atoms.info.get("multiplicity", 1)
            image.calc = type(driver)(image.copy(), dict(driver.config), verbose=driver.verbose).to_ase_calc()

    autoneb = AutoNEB(
        attach_calculators=attach_calculators,
        prefix=os.path.join(scratch_dir, filename),
        n_simul=1,
        n_max=max_images,
        iter_folder=scratch_dir,
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

    write_path_results(autoneb.all_images, filename)
    if clear_scratch and os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
