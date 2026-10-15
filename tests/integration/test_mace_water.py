import numpy as np
import pytest
from ase.units import Hartree

pytestmark = pytest.mark.mace


@pytest.fixture(scope="module")
def mace_config(mace_model_path):
    return {"model_path": mace_model_path, "model_type": "mp", "head": "omol", "precision": "float64"}


def make_driver(atoms, config, **overrides):
    from qc_launcher.drivers.mace_driver import MACEDriver

    return MACEDriver(atoms, {**config, **overrides}, verbose=False)


def test_energy_forces_hessian(water, mace_config):
    driver = make_driver(water, mace_config)
    energy = driver.compute_energy()
    assert np.isfinite(energy)
    assert driver.compute_forces().shape == (3, 3)
    hessian = driver.compute_hessian(hess_format="ase")
    assert hessian.shape == (9, 9)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-4)
    assert driver.to_pyscf_mf().e_tot == pytest.approx(energy / Hartree, rel=1e-10)
    assert water.info["spin"] == 1


def test_neighbor_list_implementations_agree(water, mace_config):
    pytest.importorskip("nvalchemiops")
    e_mace = make_driver(water, mace_config, neighbor_list="mace").compute_energy()
    e_nv = make_driver(water, mace_config, neighbor_list="nvalchemi").compute_energy()
    assert e_nv == pytest.approx(e_mace, abs=1e-8)
    big = water.copy()  # non-periodic cluster of four waters, more neighbours within the cutoff
    for shift in ((3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 3.0)):
        copy = water.copy()
        copy.translate(shift)
        big += copy
    big.info.update(water.info)
    e_mace = make_driver(big, mace_config, neighbor_list="mace").compute_energy()
    e_nv = make_driver(big, mace_config, neighbor_list="nvalchemi").compute_energy()
    assert e_nv == pytest.approx(e_mace, abs=1e-8)


def test_periodic_neighbor_list(water, mace_config):
    pytest.importorskip("nvalchemiops")
    from ase.build import bulk

    crystal = bulk("Cu", cubic=True)
    crystal.info.update(charge=0, multiplicity=1)
    e_mace = make_driver(crystal, mace_config, neighbor_list="mace").compute_energy()
    e_nv = make_driver(crystal, mace_config, neighbor_list="nvalchemi").compute_energy()
    assert e_nv == pytest.approx(e_mace, abs=1e-6)


@pytest.mark.sella
def test_short_optimization(water, mace_config, in_tmp):
    from qc_launcher.tasks import load_task

    driver = make_driver(water, mace_config)
    load_task("opt")(driver, {"max_steps": 5}, "water")
    assert (in_tmp / "water_opt.xyz").exists()
