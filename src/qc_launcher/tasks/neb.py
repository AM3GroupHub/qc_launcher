"""Nudged elastic band (optionally climbing image) with ASE's NEB and FIRE."""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE

from qc_launcher.drivers import BaseDriver

from .common import check_same_species, relax_endpoints, write_path_results


def run_neb(driver: BaseDriver, config: dict, atoms_list: Sequence[Atoms], filename: str = "molecule") -> None:
    """Config keys (defaults): ``num_images`` (len(input) - 2), ``fmax`` [eV/Angstrom] (0.05), ``climb`` (False),
    ``climb_after`` (0; NEB steps before switching to CI-NEB), ``max_steps`` (200), ``endpoint_max_steps`` (step
    limit for the endpoint relaxations; None = until converged), ``spring_constant`` (0.1), ``neb_method``
    (improvedtangent), ``fixed_reactant`` / ``fixed_product`` (False), ``interpolate_method`` (idpp),
    ``trajectory``. Two input frames are interpolated; more must be exactly ``num_images + 2`` frames.
    """
    check_same_species(atoms_list)
    num_images = int(config.get("num_images", len(atoms_list) - 2))
    fmax = float(config.get("fmax", 0.05))
    climb = bool(config.get("climb", False))
    max_opt_steps = config.get("endpoint_max_steps")
    max_neb_steps = int(config.get("max_steps", 200))
    climb_after = int(config.get("climb_after", 0))
    spring_constant = float(config.get("spring_constant", 0.1))
    neb_method = config.get("neb_method", "improvedtangent")
    fixed_reactant = bool(config.get("fixed_reactant", False))
    fixed_product = bool(config.get("fixed_product", False))
    interpolate_method = config.get("interpolate_method", "idpp")
    trajectory = config.get("trajectory", f"{filename}_neb.traj")

    if len(atoms_list) < 2:
        raise ValueError("Input must contain at least two images (initial and final geometries).")
    if len(atoms_list) == 2:
        initial, final = atoms_list
        images = [initial] + [initial.copy() for _ in range(num_images)] + [final]
        init_chain = True
    else:
        if len(atoms_list) != num_images + 2:
            raise ValueError(
                f"Number of images in input ({len(atoms_list)}) does not match num_images + 2 ({num_images + 2})."
            )
        images = list(atoms_list)
        init_chain = False

    calc = driver.to_ase_calc()
    relax_endpoints(
        images, calc, fmax=fmax, steps=max_opt_steps, fixed_reactant=fixed_reactant, fixed_product=fixed_product
    )

    neb = NEB(
        images=images,
        k=spring_constant,
        climb=False,
        remove_rotation_and_translation=True,
        method=neb_method,
        allow_shared_calculator=True,
    )
    if init_chain:
        neb.interpolate(interpolate_method)
    for image in images:
        image.calc = calc

    opt = FIRE(neb, trajectory=trajectory)
    converged = False
    stage1_steps = climb_after if climb else max_neb_steps
    if stage1_steps > 0:
        print("Starting NEB ...")
        converged = opt.run(fmax=fmax, steps=stage1_steps)
    if climb:
        print("Starting CI-NEB ...")
        neb.climb = True
        converged = opt.run(fmax=fmax, steps=max_neb_steps - stage1_steps)
    print(
        "NEB optimization converged!"
        if converged
        else "NEB optimization did not converge within the maximum number of steps."
    )

    write_path_results(neb.images, filename)
