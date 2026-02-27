import time

import h5py
import numpy as np
import ase.io
from ase import Atoms
from ase.units import Hartree, Bohr

from qc_launcher.drivers import BaseDriver

from pyscf import symm
from pyscf.hessian import thermo

from qc_launcher.utils.utils import dump_normal_mode


def run_freq(
    driver: BaseDriver,
    config: dict,
    input_name: str = "molecule",
) -> None:
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / Bohr  # convert from Angstrom to Bohr

    # record the start time
    start_time = time.time()
    
    # compute the hessian
    hessian = driver.compute_hessian(use_cache=True, hess_format="pyscf")
    end_time = time.time()
    print(f"Hessian computation completed in {end_time - start_time:.2f} seconds.")

    datafile = config.get("datafile", f"{input_name}_data.h5")
    save_hess: bool = config.get("save_hess", False)
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
    temp = config.get("temperature", 298.15)
    press = config.get("pressure", 101325)
    thermo_info = thermo.thermo(mf, freq_au, temp=temp, press=press)
    # log thermo info
    dump_normal_mode(mf.mol, freq_info)
    thermo.dump_thermo(mf.info, thermo_info)
    end_time = time.time()
    print(f"Vibrational analysis completed in {end_time - start_time:.2f} seconds.")
    # save frequencies and normal modes
    save_freq: bool = config.get("save_freq", False)
    if save_freq:
        with h5py.File(datafile, "a") as h5f:
            h5f.create_dataset("freq_wavenumber", data=freq_info["freq_wavenumber"])
            h5f.create_dataset("norm_mode", data=freq_info["norm_mode"])
    