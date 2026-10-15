import numpy as np
import pytest
from ase.units import Bohr, Hartree

from qc_launcher.drivers.base_driver import convert_hessian, to_ev, to_ev_per_ang

from .helpers import FakeDriver


def test_unit_helpers():
    assert to_ev(1.0, "Eh") == pytest.approx(Hartree)
    assert to_ev(2.5, "eV") == 2.5
    np.testing.assert_allclose(to_ev_per_ang(np.ones((2, 3)), "Eh/Bohr"), np.full((2, 3), Hartree / Bohr))
    with pytest.raises(ValueError, match="energy unit"):
        to_ev(1.0, "kcal")
    with pytest.raises(ValueError, match="forces unit"):
        to_ev_per_ang(np.ones(3), "kcal/A")


def test_convert_hessian_round_trip(rng):
    natm = 3
    h_ase = rng.normal(size=(9, 9))
    h_ase = h_ase + h_ase.T
    h_pyscf = convert_hessian(h_ase, natm, "pyscf")
    assert h_pyscf.shape == (natm, natm, 3, 3)
    np.testing.assert_allclose(h_pyscf[1, 2], h_ase[3:6, 6:9] * Bohr**2 / Hartree)
    np.testing.assert_allclose(convert_hessian(h_pyscf, natm, "ase"), h_ase)
    assert convert_hessian(h_ase, natm, "ase") is h_ase
    with pytest.raises(ValueError, match="shape"):
        convert_hessian(h_ase, 4, "ase")
    with pytest.raises(ValueError, match="hess_format"):
        convert_hessian(h_ase, natm, "orca")


def test_hessian_cache_returns_private_copies(water):
    driver = FakeDriver(water, {"k": 2.0})
    h1 = driver.compute_hessian(hess_format="ase")
    h1[:] = 0.0
    h2 = driver.compute_hessian(hess_format="ase")
    assert driver.hessian_calls == 1
    np.testing.assert_allclose(h2, 2.0 * np.eye(9))
    h3 = driver.compute_hessian(hess_format="pyscf")
    assert h3.shape == (3, 3, 3, 3) and driver.hessian_calls == 1
    water.positions[0, 0] += 0.01
    driver.compute_hessian()
    assert driver.hessian_calls == 2
    driver.clear_cache()
    driver.compute_hessian()
    assert driver.hessian_calls == 3


def test_positions_changed_tracks_in_place_mutation(water):
    driver = FakeDriver(water)
    assert driver._positions_changed()
    driver._mark_positions()
    assert not driver._positions_changed()
    water.positions[1, 2] += 1e-3  # same object, mutated in place (what Sella does)
    assert driver._positions_changed()
    driver._mark_positions()
    driver.update_atoms(water.copy())  # a different object with equal positions
    assert not driver._positions_changed()


def test_update_atoms_rejects_different_system(water):
    driver = FakeDriver(water)
    other = water.copy()
    other.info["charge"] = 1
    with pytest.raises(ValueError, match="Charge"):
        driver.update_atoms(other)
    other = water.copy()
    other.set_atomic_numbers([8, 1, 2])
    with pytest.raises(ValueError, match="Atomic numbers"):
        driver.update_atoms(other)
    driver.update_atoms(None)
    assert driver.atoms is water
    with pytest.raises(ValueError, match="No structure"):
        FakeDriver(None).update_atoms(None)


def test_energy_and_forces_through_public_api(water):
    driver = FakeDriver(water, {"k": 3.0})
    pos = water.get_positions()
    assert driver.compute_energy() == pytest.approx(1.5 * (pos**2).sum())
    np.testing.assert_allclose(driver.compute_forces(), -3.0 * pos)
