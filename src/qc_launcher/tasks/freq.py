import time

import h5py
import numpy as np
from ase.units import Bohr

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
    start_time = time.time()
    
    # get config
    datafile = config.get("datafile", f"{filename}_data.h5")
    temperature = config.get("temperature", 298.15)
    pressure = config.get("pressure", 101325)
    vibfile = config.get("vibfile", f"{filename}_vib.txt")
    save_hess: bool = config.get("save_hess", False)
    save_freq: bool = config.get("save_freq", False)
    qrrho: bool = config.get("qrrho", True)
    alpha: float = config.get("qrrho_alpha", 4.0)
    omega0: float = config.get("qrrho_omega_cutoff", 100.0)

    # compute the hessian
    hessian = driver.compute_hessian(use_cache=True, hess_format="pyscf")
    end_time = time.time()
    print(f"Hessian computation completed in {end_time - start_time:.2f} seconds.")

    # save hessian
    if save_hess:
        with h5py.File(datafile, "a") as h5f:
            h5f.create_dataset("hessian", data=hessian)
    
    # vibrational analysis
    start_time = time.time()
    mf = driver.to_pyscf_mf()
    freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
    # imaginary frequencies
    freq_au = freq_info["freq_au"]
    num_imag = np.sum(freq_au < 0)
    if num_imag > 0:
        print(f"Note: {num_imag} imaginary frequencies detected!")
    thermo_info = thermo.thermo(mf, freq_au, temperature=temperature, pressure=pressure)
    # log thermo info
    dump_normal_mode(mf.mol, freq_info)
    thermo.dump_thermo(mf.mol, thermo_info)
    if qrrho:
        qrrho_info = qrrho_thermo(
            thermo_info=thermo_info,
            freq=freq_au,
            temperature=temperature,
            omega0=omega0,
            alpha=alpha,
        )
        print("\n============== quasi-RRHO correction =============")
        print(f"RRHO  S_vib [Eh/K]    : {thermo_info['S_vib'][0]:16.10e}")
        print(f"qRRHO S_vib [Eh/K]    : {qrrho_info['S_vib_qrrho'][0]:16.10e}")
        print(f"RRHO  H_vib [Eh]      : {thermo_info['H_vib'][0]:16.10f}")
        print(f"qRRHO H_vib [Eh]      : {qrrho_info['H_vib_qrrho'][0]:16.10f}")
        print(f"RRHO  G_tot [Eh]      : {thermo_info['G_tot'][0]:16.10f}")
        print(f"qRRHO G_tot [Eh]      : {qrrho_info['G_tot_qrrho'][0]:16.10f}")
        print("==================================================\n")
    
    write_pyvibms(vibfile, driver.atoms.get_chemical_symbols(),
        freq_info["freq_wavenumber"], freq_info["norm_mode"]
    )
    end_time = time.time()
    print(f"Vibrational analysis completed in {end_time - start_time:.2f} seconds.")
    
    # save frequencies and normal modes
    if save_freq:
        with h5py.File(datafile, "a") as h5f:
            h5f.create_dataset("freq_wavenumber", data=freq_info["freq_wavenumber"])
            h5f.create_dataset("norm_mode", data=freq_info["norm_mode"])
    