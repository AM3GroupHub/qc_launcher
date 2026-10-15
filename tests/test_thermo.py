"""Unit tests for the quasi-RRHO thermochemistry in qc_launcher.utils.utils.qrrho_thermo.

The golden numbers were captured from the implementation before the refactor (2026-10-16) and guard the
numerics. The independent reimplementation is written straight from the published equations:
Grimme, Chem. Eur. J. 2012, 18, 9955 (entropy) and Li et al., J. Phys. Chem. C 2015, 119, 1840 (enthalpy).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf.data import nist

from qc_launcher.utils.utils import qrrho_thermo

pytestmark = pytest.mark.pyscf

T = 298.15
FREQ_CM1 = np.array([30.0, 80.0, 150.0, 500.0, 1500.0, 3500.0])
AU2HZ = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2)) ** 0.5 / (2 * np.pi)
C_CM = nist.LIGHT_SPEED_SI * 100.0
FREQ_AU = FREQ_CM1 * C_CM / AU2HZ

STUB = {
    "S_elec": (0.0, "Eh/K"),
    "S_trans": (1.0e-5, "Eh/K"),
    "S_rot": (2.0e-5, "Eh/K"),
    "S_vib": (0.0, "Eh/K"),
    "H_elec": (-1.0, "Eh"),
    "H_trans": (2.0e-3, "Eh"),
    "H_rot": (1.0e-3, "Eh"),
    "H_vib": (0.0, "Eh"),
    "G_elec": (-1.0, "Eh"),
    "G_trans": (-1.0e-3, "Eh"),
    "G_rot": (-5.0e-3, "Eh"),
    "G_vib": (0.0, "Eh"),
    "G_tot": (0.0, "Eh"),
}

GOLDEN = {
    1.0e-44: {
        "S_vib_qrrho": 1.702665192376536e-05,
        "H_vib_qrrho": 0.014746466898221833,
        "G_vib_qrrho": 0.009669970627151192,
        "S_tot_qrrho": 4.7026651923765365e-05,
        "H_tot_qrrho": -0.9822535331017782,
        "G_tot_qrrho": -0.9963300293728488,
    },
    3.0e-46: {
        "S_vib_qrrho": 1.6965799632077032e-05,
        "H_vib_qrrho": 0.014746466898221833,
        "G_vib_qrrho": 0.009688113737918067,
        "S_tot_qrrho": 4.696579963207703e-05,
        "H_tot_qrrho": -0.9822535331017782,
        "G_tot_qrrho": -0.9963118862620819,
    },
}


def reference_qrrho(freq_cm1, temperature, omega0=100.0, alpha=4.0, b_av=1.0e-44):
    """Independent implementation of Grimme's entropy and the Head-Gordon enthalpy interpolation."""
    h, kB = nist.PLANCK, nist.BOLTZMANN
    R = kB / nist.HARTREE2J  # gas constant in Eh/K per molecule
    s_sum, h_sum = 0.0, 0.0
    for w_cm in freq_cm1:
        nu = w_cm * C_CM  # Hz
        theta = h * nu / (kB * temperature)
        s_ho = theta / (np.exp(theta) - 1.0) - np.log(1.0 - np.exp(-theta))
        u_ho = theta / 2.0 + theta / (np.exp(theta) - 1.0)
        mu = h / (8.0 * np.pi**2 * nu)
        mu_eff = mu * b_av / (mu + b_av)
        s_fr = 0.5 + np.log(np.sqrt(8.0 * np.pi**3 * mu_eff * kB * temperature / h**2))
        weight = 1.0 / (1.0 + (omega0 / w_cm) ** alpha)
        s_sum += weight * s_ho + (1.0 - weight) * s_fr
        h_sum += weight * u_ho + (1.0 - weight) * 0.5
    return R * s_sum, R * temperature * h_sum


@pytest.mark.parametrize("b_av", sorted(GOLDEN))
def test_qrrho_golden(b_av):
    result = qrrho_thermo(STUB, FREQ_AU, temperature=T, omega0=100.0, alpha=4.0, b_av=b_av)
    for key, expected in GOLDEN[b_av].items():
        assert result[key][0] == pytest.approx(expected, rel=1e-12), key


@pytest.mark.parametrize("b_av", [1.0e-44, 3.0e-46])
def test_qrrho_matches_independent_formula(b_av):
    result = qrrho_thermo(STUB, FREQ_AU, temperature=T, b_av=b_av)
    s_ref, h_ref = reference_qrrho(FREQ_CM1, T, b_av=b_av)
    assert result["S_vib_qrrho"][0] == pytest.approx(s_ref, rel=1e-10)
    assert result["H_vib_qrrho"][0] == pytest.approx(h_ref, rel=1e-10)
    assert result["G_vib_qrrho"][0] == pytest.approx(h_ref - T * s_ref, rel=1e-10)


def test_imaginary_modes_are_dropped():
    freq = np.concatenate([[-FREQ_AU[0]], FREQ_AU[1:]])
    with_imag = qrrho_thermo(STUB, freq, temperature=T)
    without = qrrho_thermo(STUB, FREQ_AU[1:], temperature=T)
    assert with_imag["S_vib_qrrho"][0] == pytest.approx(without["S_vib_qrrho"][0], rel=1e-14)


def test_zero_reference_frequency_reduces_to_rrho_with_pyscf():
    """With omega0 = 0 the damping weight is exactly 1, so qRRHO must equal pyscf's RRHO values."""
    from pyscf import gto
    from pyscf.hessian import thermo

    mol = gto.M(atom="O 0 0 0.1174; H 0 0.7572 -0.4696; H 0 -0.7572 -0.4696", basis="sto-3g", verbose=0)
    mf = SimpleNamespace(mol=mol, e_tot=-75.0)
    freq_au = np.array([1600.0, 3650.0, 3750.0]) * C_CM / AU2HZ
    info = thermo.thermo(mf, freq_au, temperature=T, pressure=101325)
    result = qrrho_thermo(info, freq_au, temperature=T, omega0=0.0)
    assert result["S_vib_qrrho"][0] == pytest.approx(info["S_vib"][0], rel=1e-12)
    assert result["H_vib_qrrho"][0] == pytest.approx(info["H_vib"][0], rel=1e-12)
    assert result["G_tot_qrrho"][0] == pytest.approx(info["G_tot"][0], rel=1e-12)
    # totals are internally consistent
    assert result["G_tot_qrrho"][0] == pytest.approx(result["H_tot_qrrho"][0] - T * result["S_tot_qrrho"][0], rel=1e-12)
