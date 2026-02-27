import os
from typing import Dict

import numpy as np
from pyscf import gto, lib


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
