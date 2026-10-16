"""Harmonic frequencies and RRHO / quasi-RRHO thermochemistry on top of pyscf's thermo module."""

from __future__ import annotations

import json

import numpy as np
from ase.units import Bohr, _Nav
from ase.units import m as meter
from pyscf import symm
from pyscf.hessian import thermo

from qc_launcher.drivers import BaseDriver
from qc_launcher.drivers.base_driver import convert_hessian
from qc_launcher.utils.utils import dump_normal_mode, qrrho_thermo, write_pyvibms

from .common import timed

AMU_BOHR2_TO_KG_M2 = 1e-3 * Bohr**2 / (_Nav * meter**2)


def _rotor_type(rot_const: np.ndarray) -> str:
    """Same thresholds as pyscf.hessian.thermo (rotational constants in GHz)."""
    if np.all(rot_const > 1e8):
        return "ATOM"
    if rot_const[0] > 1e8 and rot_const[1] - rot_const[2] < 1e-3:
        return "LINEAR"
    return "REGULAR"


def resolve_bav(value, mol, rot_const: np.ndarray) -> float:
    """Average moment of inertia for the free-rotor entropy, in kg m^2.

    ``value`` is a number (Grimme's default is 1e-44) or ``"auto"`` for the isotropic average Tr(I)/3 of the
    molecule itself (Tr(I)/2 for linear molecules), which is what ORCA uses.
    """
    if isinstance(value, str) and value.lower() == "auto":
        coords = mol.atom_coords()
        mass = mol.atom_mass_list(isotope_avg=True)
        coords = coords - (coords * mass[:, None]).sum(axis=0) / mass.sum()
        trace = 2.0 * np.sum(mass * (coords**2).sum(axis=1))  # amu Bohr^2
        rotor = _rotor_type(np.asarray(rot_const))
        b_av = {"ATOM": 0.0, "LINEAR": trace / 2.0}.get(rotor, trace / 3.0) * AMU_BOHR2_TO_KG_M2
    else:
        try:
            b_av = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid Bav {value!r}: use a number in kg m^2 or 'auto'.") from None
    if b_av < 1e-50:
        print("Warning: Bav is very small, setting to 1e-50 to avoid numerical issues.")
        b_av = 1e-50
    return b_av


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_thermo_json(path: str, results: dict, frequencies_cm1: np.ndarray, qrrho_params: dict | None) -> None:
    payload = {
        key: {"value": _jsonable(val[0]), "unit": val[1]}
        if isinstance(val, tuple) and len(val) == 2
        else _jsonable(val)
        for key, val in results.items()
    }
    payload["frequencies_cm1"] = np.real(frequencies_cm1).tolist()
    payload["qrrho_parameters"] = qrrho_params
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_freq(driver: BaseDriver, config: dict, filename: str = "molecule") -> dict:
    """Config keys (defaults): ``temperature`` [K] (298.15), ``pressure`` [Pa] (101325), ``vibfile``,
    ``save_hess`` (False), ``symm_geom_tol`` [Angstrom] (pyscf symmetry detection tolerance), ``qrrho`` (True),
    ``alpha`` (4), ``omega0`` [cm^-1] (100), ``Bav`` [kg m^2] (1e-44, or "auto").

    Returns pyscf's thermo dict merged with the ``*_qrrho`` keys and writes ``{filename}_thermo.json``.
    Imaginary modes are reported but excluded from the thermochemistry.
    """
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = float(config["symm_geom_tol"]) / Bohr  # process-wide pyscf setting
    temperature = float(config.get("temperature", 298.15))
    pressure = float(config.get("pressure", 101325))
    vibfile = config.get("vibfile", f"{filename}_vib.txt")
    qrrho = bool(config.get("qrrho", True))
    alpha = float(config.get("alpha", 4.0))
    omega0 = float(config.get("omega0", 100.0))
    bav_setting = config.get("Bav", 1.0e-44)
    save_hess = bool(config.get("save_hess", False))

    with timed("Hessian computation"):
        hessian = driver.compute_hessian(use_cache=True, hess_format="pyscf")
    if save_hess:
        np.savetxt(f"{filename}_hess.txt", convert_hessian(hessian, len(driver.atoms), "ase"))

    with timed("Vibrational analysis"):
        mf = driver.to_pyscf_mf()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
        freq_au = freq_info["freq_au"]
        num_imag = int(np.sum(freq_au < 0))
        if num_imag > 0:
            print(f"Note: {num_imag} imaginary frequencies detected!")
        thermo_info = thermo.thermo(mf, freq_au, temperature=temperature, pressure=pressure)
        dump_normal_mode(mf.mol, freq_info)
        thermo.dump_thermo(mf.mol, thermo_info)

        results = dict(thermo_info)
        qrrho_params = None
        if qrrho:
            b_av = resolve_bav(bav_setting, mf.mol, thermo_info["rot_const"][0])
            qrrho_params = {"alpha": alpha, "omega0_cm1": omega0, "Bav_kg_m2": b_av}
            results.update(qrrho_thermo(thermo_info, freq_au, temperature, omega0, alpha, b_av))
            print("\n============== quasi-RRHO correction =============")
            print(f"alpha = {alpha:g}, omega0 = {omega0:g} cm^-1, Bav = {b_av:.4e} kg m^2")
            for key, unit, fmt in (
                ("S_vib", "Eh/K", "16.10e"),
                ("S_tot", "Eh/K", "16.10e"),
                ("H_vib", "Eh", "16.10f"),
                ("H_tot", "Eh", "16.10f"),
                ("G_tot", "Eh", "16.10f"),
            ):
                print(f"RRHO  {key} [{unit}]{' ' * (7 - len(unit))}: {results[key][0]:{fmt}}")
                print(f"qRRHO {key} [{unit}]{' ' * (7 - len(unit))}: {results[key + '_qrrho'][0]:{fmt}}")
            print("Note: qRRHO entropy follows Grimme (2012); qRRHO enthalpy applies the Head-Gordon (2015)")
            print("      damping to the ZPE-inclusive vibrational energy. ZPE and E_0K above are RRHO values.")
            print("==================================================\n")

        write_pyvibms(
            vibfile, driver.atoms.get_chemical_symbols(), freq_info["freq_wavenumber"], freq_info["norm_mode"]
        )
        write_thermo_json(f"{filename}_thermo.json", results, freq_info["freq_wavenumber"], qrrho_params)
    return results
