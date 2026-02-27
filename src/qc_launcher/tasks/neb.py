import numpy as np
import ase.io
from ase.mep import NEB
from ase.optimize import FIRE

from qc_launcher.drivers.base_driver import BaseDriver


def run_neb(
    driver: BaseDriver,
    config: dict,
    atoms_list: list,
    filename: str = "molecule",
) -> BaseDriver:

    # check if all images have the same atoms
    symbols_list = [atoms.symbols for atoms in atoms_list]
    if not all(np.array_equal(symbols, symbols_list[0]) for symbols in symbols_list):
        raise ValueError("All images must have the same atoms (same symbols in the same order).")


    # get config
    num_images = config.get("num_images", len(atoms_list) - 2)
    fmax = float(config.get("fmax", 0.05))
    climb = config.get("climb", False)
    steps = config.get("steps", 1000)
    climb_after = config.get("climb_after", 0)
    spring_constant = float(config.get("spring_constant", 0.1))
    neb_method = config.get("neb_method", "improved_tangent")
    opt_termial: bool = config.get("opt_terminal", True)
    interpolate_method: str = config.get("interpolate_method", "idpp")
    trajectory = config.get("trajectory", f"{filename}_neb.traj")


    if len(atoms_list) == 2:
        init_atoms, final_atoms = atoms_list[0], atoms_list[1]
        images = [init_atoms]
        for _ in range(num_images):
            images.append(init_atoms.copy())
        images.append(final_atoms)
        init_chain = True
    elif len(atoms_list) > 2:
        if len(atoms_list) != num_images + 2:
            raise ValueError(f"Number of images in input ({len(atoms_list)}) does not match num_images + 2 ({num_images + 2}).")
        images = atoms_list
        init_chain = False
    else:
        raise ValueError("Input must contain at least two images (initial and final geometries).")

    # get calc
    calc = driver.to_ase_calc()
    
    # optimize terminal images first
    if opt_termial and init_chain:
        images[0].calc = calc
        images[-1].calc = calc
        print("Optimizing initial image...")
        with FIRE(images[0]) as opt:
            opt.run(fmax=fmax)
        print("Optimizing final image...")
        with FIRE(images[-1]) as opt:
            opt.run(fmax=fmax)
    
    # set up NEB and interpolate if needed
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
    
    # set up calculators for all images
    for image in images:
        image.calc = calc

    # set up optimizer and run NEB
    opt = FIRE(neb, trajectory=trajectory)
    # stage 1: regular NEB
    stage1_steps = climb_after if climb else steps
    if stage1_steps > 0:
        print("Starting NEB ...")
        opt.run(fmax=fmax, steps=stage1_steps)
    # stage 2: CI-NEB if requested
    if climb:
        print("Starting CI-NEB ...")
        neb.climb = True
        opt.run(fmax=fmax, steps=steps-stage1_steps)
    images = neb.images
    
    # chekc convergence
    if opt.converged():
        print("NEB optimization converged!")
    else:
        print("NEB optimization did not converge within the maximum number of steps.")
    
    # save final images
    ase.io.write(f"{filename}_neb.xyz", images)
    # print energies of all images
    energies = np.array([image.get_potential_energy() for image in images])
    for i, energy in enumerate(energies):
        print(f"Image {i:02d}: Energy = {energy:.6f} eV")
