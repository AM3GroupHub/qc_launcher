"""Minimum and transition-state optimization with geomeTRIC (pyscf's optimizer), same config keys as the Sella path.

geomeTRIC drives the driver's ASE calculator through a custom engine (energies in Hartree, gradients in
Hartree/Bohr); the exact Hessian of the driver is handed over as a Hessian file when ``calc_hess`` is set.
"""

from __future__ import annotations

import re
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Bohr, Hartree

from qc_launcher.drivers import BaseDriver
from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

from .common import FMAX_DEFAULT, timed

COORDSYS = ("tric", "cart", "prim", "dlc", "hdlc", "tric-p")
CONSTRAINT_KEYWORDS = {"fix_bond": "distance", "fix_angle": "angle", "fix_dihedral": "dihedral"}
# geomeTRIC option names that are passed straight through when present in the config
PASSTHROUGH_KEYS = (
    "trust",
    "tmax",
    "tmin",
    "enforce",
    "epsilon",
    "reset",
    "qccnv",
    "molcnv",
    "subfrctor",
    "convergence_cmax",
)


def geometric_constraints(cons_dict: dict | None) -> str | None:
    """Translate ``{fix_translation: [i], fix_bond: [[i, j]], fix_angle: [[i, j, k]], fix_dihedral: [[i, j, k, l]]}``
    (0-based) into geomeTRIC's ``$freeze`` block (1-based); None when nothing is constrained."""
    if not cons_dict:
        return None
    unknown = set(cons_dict) - set(CONSTRAINT_KEYWORDS) - {"fix_translation"}
    if unknown:
        raise ValueError(
            f"Unknown constraint types {sorted(unknown)}. Use {('fix_translation', *CONSTRAINT_KEYWORDS)}."
        )
    lines = ["$freeze"]
    for index in cons_dict.get("fix_translation") or []:
        lines.append(f"xyz {int(index) + 1}")
    for name, keyword in CONSTRAINT_KEYWORDS.items():
        for item in cons_dict.get(name) or []:
            lines.append(f"{keyword} " + " ".join(str(int(i) + 1) for i in item))
    return "\n".join(lines) + "\n" if len(lines) > 1 else None


def convergence_kwargs(config: dict) -> dict:
    """Gaussian-style criteria in ASE units (``ediff`` eV, ``fmax`` / ``frms`` eV/Angstrom, ``dmax`` / ``drms``
    Angstrom) as geomeTRIC's atomic-unit options. Keys the config does not set follow ``convergence_set`` (GAU)."""
    kwargs: dict = {}
    if "convergence_set" in config:
        kwargs["convergence_set"] = str(config["convergence_set"])
    else:
        kwargs.update(
            convergence_energy=1e-6,
            convergence_gmax=FMAX_DEFAULT * Bohr / Hartree,
            convergence_grms=3.0e-4,
            convergence_dmax=1.8e-3,
            convergence_drms=1.2e-3,
        )
    if "ediff" in config:
        kwargs["convergence_energy"] = float(config["ediff"]) / Hartree
    if "fmax" in config:
        kwargs["convergence_gmax"] = float(config["fmax"]) * Bohr / Hartree
    if "frms" in config:
        kwargs["convergence_grms"] = float(config["frms"]) * Bohr / Hartree
    if "dmax" in config:
        kwargs["convergence_dmax"] = float(config["dmax"])
    if "drms" in config:
        kwargs["convergence_drms"] = float(config["drms"])
    return kwargs


def write_hessian_file(driver: BaseDriver, atoms: Atoms, path: Path) -> Path:
    """The driver's exact Cartesian Hessian in geomeTRIC's text format (Hartree/Bohr^2, 3N x 3N)."""
    hessian = np.asarray(driver.compute_hessian(atoms, use_cache=True, hess_format="ase"), dtype=float)
    hessian = hessian.reshape(3 * len(atoms), 3 * len(atoms)) * Bohr**2 / Hartree
    np.savetxt(path, hessian)
    return path


def _import_geometric():
    try:
        import geometric.engine
        import geometric.molecule
        import geometric.optimize
        from geometric.errors import GeomOptNotConvergedError
    except ModuleNotFoundError as exc:
        if is_missing_package(exc, "geometric"):
            raise missing_optional_dependency("The geomeTRIC optimizer", "geometric", "geometric") from exc
        raise
    return geometric, GeomOptNotConvergedError


def make_engine(driver: BaseDriver, atoms: Atoms):
    """geomeTRIC engine evaluating the driver's ASE calculator on the driver's own Atoms (keeps charge/multiplicity)."""
    geometric, _ = _import_geometric()

    class DriverEngine(geometric.engine.Engine):
        def __init__(self):
            molecule = geometric.molecule.Molecule()
            molecule.elem = atoms.get_chemical_symbols()
            molecule.xyzs = [atoms.get_positions().copy()]
            super().__init__(molecule)
            self.atoms = atoms
            self.calculator = driver.to_ase_calc()
            self.atoms.calc = self.calculator
            self.cycle = 0
            self.last = None  # (energy eV, forces eV/Angstrom) of the latest evaluation

        def calc_new(self, coords, dirname):
            self.atoms.set_positions(coords.reshape(-1, 3) * Bohr)
            energy = float(self.atoms.get_potential_energy())
            forces = np.asarray(self.atoms.get_forces(), dtype=float)
            self.cycle += 1
            self.last = (energy, forces)
            print(f"geomeTRIC evaluation {self.cycle}: E = {energy:.8f} eV, max |F| = {np.abs(forces).max():.6f} eV/Å")
            return {"energy": energy / Hartree, "gradient": (-forces / Hartree * Bohr).ravel()}

        def copy_scratch(self, src, dest):
            return

    return DriverEngine()


def _frames_from_xyz(path: Path) -> tuple[list[np.ndarray], list[float | None]]:
    """Frames and energies (Eh, parsed from the comment lines) of a geomeTRIC _optim.xyz file."""
    lines = path.read_text().splitlines()
    frames, energies = [], []
    i = 0
    while i < len(lines) and lines[i].strip():
        natoms = int(lines[i])
        match = re.search(r"Energy\s+(-?\d+\.\d+)", lines[i + 1])
        energies.append(float(match.group(1)) if match else None)
        block = lines[i + 2 : i + 2 + natoms]
        frames.append(np.array([[float(x) for x in line.split()[1:4]] for line in block]))
        i += 2 + natoms
    return frames, energies


def run_opt_geometric(driver: BaseDriver, config: dict, filename: str = "molecule") -> bool:
    """Config keys (defaults): ``ts`` (False), ``calc_hess`` (= ts; the driver's exact Hessian is given to geomeTRIC
    as a file), ``hessian`` (geomeTRIC's never | first | each | last | first+last | stop, used when the driver's
    Hessian is not requested), ``coordsys`` (tric | cart | prim | dlc | hdlc), ``trajectory``, ``outputfile``,
    ``prefix`` ({filename}_geometric: log, _optim.xyz and .tmp directory), ``constraints`` (same dictionary as the
    Sella path), ``max_steps`` (150), ``convergence_set`` (GAU) and the Gaussian criteria ``ediff`` [eV],
    ``fmax`` / ``frms`` [eV/Angstrom], ``dmax`` / ``drms`` [Angstrom]; geomeTRIC's ``trust``, ``tmax``, ``tmin``,
    ``enforce``, ``epsilon``, ``reset``, ``qccnv``, ``molcnv``, ``subfrctor``, ``convergence_cmax`` pass through.

    Returns True when geomeTRIC reports convergence. The driver's Atoms hold the final geometry.
    """
    geometric, GeomOptNotConvergedError = _import_geometric()
    with timed("Optimization (geomeTRIC)"):
        atoms = driver.atoms
        optts = bool(config.get("ts", False))
        calc_hess = bool(config.get("calc_hess", optts))
        coordsys = str(config.get("coordsys", "tric")).lower()
        if coordsys not in COORDSYS:
            raise ValueError(f"Unknown coordsys '{coordsys}'. Use one of {COORDSYS}.")
        trajectory = config.get("trajectory", f"{filename}_opt.traj")
        opt_outputfile = config.get("outputfile", f"{filename}_opt.xyz")
        prefix = str(config.get("prefix", f"{filename}_geometric"))
        max_steps = int(config.get("max_steps", 150))

        kwargs = dict(
            coordsys=coordsys,
            transition=optts,
            maxiter=max_steps,
            prefix=prefix,
            input=f"{prefix}.xyz",  # only used by geomeTRIC to derive names; no file is read
            **convergence_kwargs(config),
        )
        for key in PASSTHROUGH_KEYS:
            if key in config:
                kwargs[key] = config[key]
        constraints = geometric_constraints(config.get("constraints"))
        if constraints:
            if coordsys in ("cart", "prim", "tric-p"):
                raise ValueError("geomeTRIC constraints need a delocalized coordinate system (tric, dlc or hdlc).")
            constraints_path = Path(f"{prefix}_constraints.txt")
            constraints_path.write_text(constraints)
            kwargs["constraints"] = str(constraints_path)
            print(f"Applied constraints:\n{constraints.strip()}")
        if calc_hess:
            hessian_path = write_hessian_file(driver, atoms, Path(f"{prefix}_hessian.txt"))
            kwargs["hessian"] = f"file:{hessian_path}"
            print(f"Exact Hessian of the driver written to {hessian_path} for geomeTRIC.")
        elif config.get("hessian") is not None:
            kwargs["hessian"] = str(config["hessian"])

        engine = make_engine(driver, atoms)
        converged = True
        try:
            progress = geometric.optimize.run_optimizer(customengine=engine, **kwargs)
            frames = [np.asarray(x) for x in progress.xyzs]
            energies = (
                [float(e) for e in progress.qm_energies]
                if len(progress.qm_energies) == len(frames)
                else [None] * len(frames)
            )
        except GeomOptNotConvergedError:
            converged = False
            optim_xyz = Path(f"{prefix}_optim.xyz")
            frames, energies = _frames_from_xyz(optim_xyz) if optim_xyz.exists() else ([atoms.get_positions()], [None])

        atoms.set_positions(frames[-1])
        images = []
        for positions, energy in zip(frames, energies):
            image = atoms.copy()
            image.set_positions(positions)
            if energy is not None:
                image.calc = SinglePointCalculator(image, energy=energy * Hartree)
            images.append(image)
        ase.io.write(trajectory, images)
        ase.io.write(opt_outputfile, atoms, columns=["symbols", "positions"])

        if converged:
            print(f"Optimization converged! ({len(frames) - 1} geomeTRIC steps)")
        else:
            print("Optimization did not converge within the maximum number of steps.")
            if engine.last is not None:
                energy, forces = engine.last
                print(f"Final energy          : {energy:.8f} eV")
                print(f"Final MAX force       : {np.max(np.linalg.norm(forces, axis=1)):.6e} eV/Å")
                print(f"Final RMS force       : {np.sqrt(np.mean(forces**2)):.6e} eV/Å")
        return converged
