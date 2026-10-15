import numpy as np
import pytest
from ase.units import Bohr, Hartree

pytestmark = pytest.mark.tblite

CONFIG = {"method": "GFN2-xTB", "verbosity": 0}


def make_driver(atoms, **overrides):
    from qc_launcher.drivers.tblite_driver import TBLiteDriver

    return TBLiteDriver(atoms, {**CONFIG, **overrides}, verbose=False)


def test_energy_matches_direct_tblite(water):
    from tblite.interface import Calculator

    xtb = Calculator("GFN2-xTB", water.get_atomic_numbers(), water.get_positions() / Bohr)
    xtb.set("verbosity", 0)
    reference = xtb.singlepoint()["energy"]
    assert make_driver(water).compute_energy() / Hartree == pytest.approx(reference, abs=1e-9)


def test_driver_and_ase_calculator_agree(water):
    driver = make_driver(water)
    atoms = water.copy()
    atoms.calc = driver.to_ase_calc()
    assert atoms.get_potential_energy() == pytest.approx(driver.compute_energy(), abs=1e-8)
    np.testing.assert_allclose(atoms.get_forces(), driver.compute_forces(), atol=1e-7)


def test_result_cache_follows_positions(water):
    driver = make_driver(water)
    first = driver.run_kernel()
    e_first = first["energy"]
    assert driver.run_kernel() is first  # unchanged positions reuse the cached result
    assert not driver._positions_changed()
    water.positions[0, 0] += 0.05  # tblite updates the Result in place, so compare energies
    assert driver._positions_changed()
    assert driver.run_kernel()["energy"] != pytest.approx(e_first, abs=1e-6)
    assert not driver._positions_changed()


def test_hessian_is_symmetric_and_matches_ase_differences(water):
    driver = make_driver(water)
    hessian = driver.compute_hessian(hess_format="ase")
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)
    np.testing.assert_allclose(driver.atoms.get_positions(), water.get_positions())  # geometry restored
    # independent two-point differences of the ASE forces
    eps = 1e-3
    atoms = water.copy()
    atoms.calc = make_driver(atoms).to_ase_calc()
    column = np.zeros(9)
    for sign in (1, -1):
        atoms.positions[1, 2] += sign * eps
        column += -sign * atoms.get_forces().ravel() / (2 * eps)
        atoms.positions[1, 2] -= sign * eps
    np.testing.assert_allclose(hessian[:, 5], column, atol=5e-2)  # both numerical; SCC accuracy limits agreement


def test_scc_failure_raises(water):
    with pytest.raises(RuntimeError, match="did not converge"):
        make_driver(water, max_iter=1).compute_energy()


@pytest.mark.sella
def test_opt_then_freq(water, in_tmp):
    from qc_launcher.tasks import load_task

    driver = make_driver(water)
    load_task("opt")(driver, {"max_steps": 50}, "water")
    np.testing.assert_allclose(driver.compute_forces(), 0.0, atol=2e-2)
    load_task("freq")(driver, {"qrrho": True}, "water")
    header = (in_tmp / "water_vib.txt").read_text().splitlines()[0]
    assert header == "3 3"  # 3 atoms, 3 vibrational modes
