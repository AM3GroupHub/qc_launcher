import os
from typing import Dict
from copy import deepcopy

import numpy as np
from pyscf import gto, lib
from pyscf.data import nist


def read_pcm_eps() -> Dict[str, float]:
    # from https://gaussian.com/scrf/
    pcm_eps_txt = os.path.join(os.path.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def dump_normal_mode(mol: gto.Mole, results: Dict[str, np.ndarray]) -> None:
    """
    The function in PySCF does not dump imagnary frequencies.
    We made a custom function to dump all frequencies and normal modes.
    Args:
        mol (gto.Mole): The molecule object.
        results (Dict[str, np.ndarray]): A dictionary containing frequencies and normal modes.
    """

    dump = mol.stdout.write
    freq_wn = results['freq_wavenumber']
    if np.iscomplexobj(freq_wn):
        freq_wn = freq_wn.real - abs(freq_wn.imag)
    nfreq = freq_wn.size

    r_mass = results['reduced_mass']
    force = results['force_const_dyne']
    vib_t = results['vib_temperature']
    mode = results['norm_mode']
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    def inline(q, col0, col1):
        return ''.join('%20.4f' % q[i] for i in range(col0, col1))
    def mode_inline(row, col0, col1):
        return '  '.join('%6.2f%6.2f%6.2f' % (mode[i,row,0], mode[i,row,1], mode[i,row,2])
                         for i in range(col0, col1))

    for col0, col1 in lib.prange(0, nfreq, 3):
        dump('Mode              %s\n' % ''.join('%20d'%i for i in range(col0,col1)))
        dump('Irrep\n')
        dump('Freq [cm^-1]          %s\n' % inline(freq_wn, col0, col1))
        dump('Reduced mass [au]     %s\n' % inline(r_mass, col0, col1))
        dump('Force const [Dyne/A]  %s\n' % inline(force, col0, col1))
        dump('Char temp [K]         %s\n' % inline(vib_t, col0, col1))
        dump('Normal mode            %s\n' % ('       x     y     z'*(col1-col0)))
        for j, at in enumerate(symbols):
            dump('    %4d%4s               %s\n' % (j, at, mode_inline(j, col0, col1)))


def write_pyvibms(
    filename: str,
    symbols: list,
    freq_wavenumbers: np.ndarray,
    norm_mode: np.ndarray,
) -> None:
    n_atoms = len(symbols)
    n_modes = len(freq_wavenumbers)

    with open(filename, "w") as f:
        f.write(f"{n_atoms} {n_modes}\n\n")
        for i, (freq, mode) in enumerate(zip(freq_wavenumbers, norm_mode)):
            f.write(f"N {freq:.4f} NULL {i+1}\n")
            for m in mode.reshape(-1):
                f.write(f"{m:.4f}\n")
            if i < n_modes - 1:
                f.write("\n")
            else:
                f.write("END\n")


def qrrho_thermo(
    thermo_info: dict,
    freq: np.ndarray,
    temperature: float = 298.15,
    omega0: float = 100.0,
    alpha: float = 4.0,
    b_av: float = 1.0e-44,  # doi/10.1002/chem.201200497
):
    kB = nist.BOLTZMANN
    h = nist.PLANCK
    R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)
    c_cm = nist.LIGHT_SPEED_SI * 100.0  # cm/s
    results = deepcopy(thermo_info)  # avoid modifying the original thermo_info
    
    # prepare vibrational terms
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * np.pi)
    idx = freq.real > 0
    real_freq_au = freq.real[idx]
    
    # convert to cm^-1 to calculate the damping factor
    freq_cm1 = real_freq_au * au2hz / c_cm
    
    # convert to reduced temperature h*nu / (kB*T)
    vib_temperature = (real_freq_au * au2hz * h) / kB
    rt = vib_temperature / temperature  # reduced temperature
    exp_mrt = np.exp(-rt)  # exponential minus reduced temperature

    # calculate quasi-RRHO correction
    # enthalpy and entropy for HO (harmonic oscillator)
    s_ho = rt / np.expm1(rt) - np.log1p(-exp_mrt)  # S_vib / R
    h_ho = 0.5 * rt + rt / (np.expm1(rt))  # H_vib / (R*T) (including zero-point energy)
    
    # entropy and enthalpy for FR (free rotor)
    nu_hz = real_freq_au * au2hz
    mu_v = h / (8.0 * np.pi**2 * nu_hz)
    mu_eff = (mu_v * b_av) / (mu_v + b_av)
    
    # partition function for free rotor
    q_fr_sqr = 8.0 * np.pi**3 * mu_eff * kB * temperature / h**2
    s_fr = 0.5 * (1.0 + np.log(q_fr_sqr))
    h_fr = 0.5
    
    # Chai-Head-Gordon damping function
    w = 1.0 / (1.0 + (omega0 / freq_cm1)**alpha)
    s_qrrho = w * s_ho + (1 - w) * s_fr
    h_qrrho = w * h_ho + (1 - w) * h_fr
    
    # record qrrho results
    results['S_vib_qrrho'] = (R_Eh * np.sum(s_qrrho), 'Eh/K')
    results['H_vib_qrrho'] = (R_Eh * temperature * np.sum(h_qrrho), 'Eh')
    results['G_vib_qrrho'] = (results['H_vib_qrrho'][0] - temperature * results['S_vib_qrrho'][0], 'Eh')

    # update total thermo properties with qrrho correction
    # keep electronic, translational, and rotational contributions unchanged
    def _sum_qrrho(f_prefix):
        keys = ('elec', 'trans', 'rot')
        base = sum(results.get(f_prefix+'_'+key, (0,))[0] for key in keys)
        return base + results[f_prefix+'_vib_qrrho'][0]

    results['S_tot_qrrho'] = (_sum_qrrho('S'), 'Eh/K')
    results['H_tot_qrrho'] = (_sum_qrrho('H'), 'Eh')
    results['G_tot_qrrho'] = (_sum_qrrho('G'), 'Eh')
    return results
