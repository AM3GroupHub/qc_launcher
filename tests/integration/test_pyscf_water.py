"""PySCF driver integration tests on water (PBE/STO-3G, seconds on CPU)."""

import numpy as np
import pytest
from ase.units import Hartree

pytestmark = pytest.mark.pyscf

CPU_CONFIG = {
    "xc": "PBE",
    "basis": "sto-3g",
    "with_df": False,
    "with_gpu": False,
    "verbose": 0,
    "threads": 2,
    "scf_conv_tol": 1e-10,
}


def make_driver(atoms, **overrides):
    from qc_launcher.drivers.pyscf_driver import PySCFDriver

    return PySCFDriver(atoms, {**CPU_CONFIG, **overrides}, verbose=False)


def reference_energy(atoms):
    from pyscf import dft, gto

    mol = gto.M(
        atom=[(s, p) for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions())], basis="sto-3g", verbose=0
    )
    mf = dft.RKS(mol, xc="PBE")
    mf.conv_tol = 1e-10
    return mf.kernel()


def test_single_point_matches_direct_pyscf(water):
    assert make_driver(water).compute_energy() / Hartree == pytest.approx(reference_energy(water), abs=1e-8)


def test_scf_follows_in_place_moves(water):
    driver = make_driver(water)
    e1 = driver.compute_energy()
    water.positions[0, 0] += 0.2  # same object mutated in place, as Sella does
    e2 = driver.compute_energy()
    assert abs(e2 - e1) > 1e-3
    assert e2 / Hartree == pytest.approx(reference_energy(water), abs=1e-8)
    assert driver.to_pyscf_mf().e_tot == pytest.approx(e2 / Hartree, abs=1e-10)


def test_ase_calculator_matches_driver(water):
    driver = make_driver(water)
    atoms = water.copy()
    atoms.calc = driver.to_ase_calc()
    assert atoms.get_potential_energy() == pytest.approx(driver.compute_energy(), abs=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), driver.compute_forces(), atol=1e-6)


def test_numerical_hessian_matches_analytic(water):
    # Hartree-Fock exchange: no integration grid, so the only difference is the finite-difference error
    analytic = make_driver(water, xc="HF").compute_hessian(hess_format="ase")
    numerical_driver = make_driver(water, xc="HF", numerical_hessian=True)
    numerical = numerical_driver.compute_hessian(hess_format="ase")
    np.testing.assert_allclose(analytic, analytic.T, atol=1e-6)
    np.testing.assert_allclose(
        numerical, analytic, atol=1e-2
    )  # eV/Angstrom^2; O(h^2) error of the 5e-3 A step is ~4e-3
    # the driver is left in a consistent state at the undisplaced geometry
    assert numerical_driver.compute_energy() == pytest.approx(make_driver(water, xc="HF").compute_energy(), abs=1e-8)


@pytest.mark.gpu
def test_gpu_matches_cpu(water):
    cpu = make_driver(water, with_df=True).compute_energy()
    gpu_driver = make_driver(water, with_df=True, with_gpu=True)
    assert gpu_driver.compute_energy() == pytest.approx(cpu, abs=1e-6)
    np.testing.assert_allclose(
        gpu_driver.compute_forces(), make_driver(water, with_df=True).compute_forces(), atol=1e-5
    )
    assert gpu_driver.compute_hessian().shape == (9, 9)


@pytest.mark.sella
def test_opt_with_exact_hessians_then_single_point(water, in_tmp):
    from qc_launcher.tasks import load_task

    driver = make_driver(water)
    load_task("opt")(driver, {"calc_hess": True, "diag_every_n": 1, "max_steps": 30}, "water")
    after_opt = driver.compute_energy()
    fresh = make_driver(driver.atoms.copy()).compute_energy()
    assert after_opt == pytest.approx(fresh, abs=1e-7)
    np.testing.assert_allclose(driver.compute_forces(), 0.0, atol=2e-2)
    load_task("freq")(driver, {"qrrho": True, "Bav": "auto"}, "water")
