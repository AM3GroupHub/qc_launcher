import time
from typing import Union

import numpy as np
from ase.units import Bohr, _Nav
from ase.units import m as meter

from qc_launcher.drivers import BaseDriver

from pyscf import symm
from pyscf.hessian import thermo

from qc_launcher.utils.utils import dump_normal_mode, write_pyvibms, qrrho_thermo


def run_freq(
    driver: BaseDriver,
    config: dict,
    filename: str = "molecule",
) -> None:
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / Bohr  # convert from Angstrom to Bohr

    # record the start time
    results = {}
    start_time = time.time()
    
    # get config
    temperature = config.get("temperature", 298.15)
    pressure = config.get("pressure", 101325)
    vibfile = config.get("vibfile", f"{filename}_vib.txt")
    qrrho: bool = config.get("qrrho", True)
    alpha: float = config.get("alpha", 4.0)
    omega0: float = config.get("omega0", 100.0)
    Bav: Union[float, str] = config.get("Bav", 1.0e-44)

    # compute the hessian
    hessian = driver.compute_hessian(use_cache=True, hess_format="pyscf")
    end_time = time.time()
    print(f"Hessian computation completed in {end_time - start_time:.2f} seconds.")

    # save hessian
    results["hessian"] = (hessian, "Eh/Bohr^2")
    
    # vibrational analysis
    start_time = time.time()
    mf = driver.to_pyscf_mf()
    freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
    # imaginary frequencies
    freq_au = freq_info["freq_au"]
    num_imag = np.sum(freq_au < 0)
    if num_imag > 0:
        print(f"Note: {num_imag} imaginary frequencies detected!")

    # save frequencies and normal modes
    results["frequencies"] = (freq_info["freq_wavenumber"], "cm^-1")
    results["normal_modes"] = (freq_info["norm_mode"], "")

    # calculate and log thermo info
    thermo_info = thermo.thermo(mf, freq_au, temperature=temperature, pressure=pressure)
    dump_normal_mode(mf.mol, freq_info)
    thermo.dump_thermo(mf.mol, thermo_info)

    # save thermo info
    results.update(thermo_info)

    # apply quasi-RRHO correction if requested
    if qrrho:
        if isinstance(Bav, str) and Bav.lower() == "auto":
            # calculate Bav = Tr[I] / 3
            mol = mf.mol
            atom_coords = mol.atom_coords()
            mass = mol.atom_mass_list(isotope_avg=True)
            mass_center = np.sum(atom_coords * mass[:, None], axis=0) / np.sum(mass)
            atom_coords -= mass_center
            r_sq = np.sum(atom_coords**2, axis=1)
            tr_I = 2.0 * np.sum(mass * r_sq)
            rot_const = thermo_info["rot_const"][0]
            rotor_type = thermo._get_rotor_type(rot_const)
            if rotor_type == "ATOM":
                b_av = 0.0
            elif rotor_type == "LINEAR":
                b_av = tr_I / 2.0
            else:
                b_av = tr_I / 3.0
            # convert from g/mol*Bohr^2 to kg*m^2
            b_av *= (1e-3 * Bohr**2 / (_Nav * meter**2))
        elif isinstance(Bav, float):
            b_av = Bav
        else:
            raise ValueError("Invalid Bav value. Use a float or 'auto'.")
        if b_av < 1e-50:
            print("Warning: Bav is very small, setting to 1e-50 to avoid numerical issues.")
            b_av = 1e-50
        qrrho_info = qrrho_thermo(
            thermo_info=thermo_info,
            freq=freq_au,
            temperature=temperature,
            omega0=omega0,
            alpha=alpha,
            b_av=b_av,
        )
        print("\n============== quasi-RRHO correction =============")
        print(f"RRHO  S_vib [Eh/K]    : {thermo_info['S_vib'][0]:16.10e}")
        print(f"qRRHO S_vib [Eh/K]    : {qrrho_info['S_vib_qrrho'][0]:16.10e}")
        print(f"RRHO  H_vib [Eh]      : {thermo_info['H_vib'][0]:16.10f}")
        print(f"qRRHO H_vib [Eh]      : {qrrho_info['H_vib_qrrho'][0]:16.10f}")
        print(f"RRHO  G_tot [Eh]      : {thermo_info['G_tot'][0]:16.10f}")
        print(f"qRRHO G_tot [Eh]      : {qrrho_info['G_tot_qrrho'][0]:16.10f}")
        print("==================================================\n")
        
        # save qRRHO thermo info
        results.update(qrrho_info)

    write_pyvibms(vibfile, driver.atoms.get_chemical_symbols(),
        freq_info["freq_wavenumber"], freq_info["norm_mode"]
    )
    end_time = time.time()
    print(f"Vibrational analysis completed in {end_time - start_time:.2f} seconds.")
