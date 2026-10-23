from importlib.util import find_spec

import numpy as np
import pytest
from ase.units import Hartree

pytestmark = pytest.mark.uma


@pytest.fixture(scope="module")
def uma_config(uma_model_path):
    return {"model_path": uma_model_path, "task_name": "omol"}


def make_driver(atoms, config, **overrides):
    from qc_launcher.drivers.uma_driver import UMADriver

    return UMADriver(atoms, {**config, **overrides}, verbose=False)


def test_energy_forces_hessian(water, uma_config):
    driver = make_driver(water, uma_config)
    energy = driver.compute_energy()
    assert np.isfinite(energy)
    forces = driver.compute_forces()
    assert forces.shape == (3, 3)
    hessian = driver.compute_hessian(hess_format="ase")
    assert hessian.shape == (9, 9)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-8)
    # one column against two-point differences of the calculator forces
    eps = 5e-3
    atoms = water.copy()
    atoms.calc = driver.to_ase_calc()
    column = np.zeros(9)
    for sign in (1, -1):
        atoms.positions[1, 2] += sign * eps
        column += -sign * atoms.get_forces().ravel() / (2 * eps)
        atoms.positions[1, 2] -= sign * eps
    np.testing.assert_allclose(hessian[:, 5], column, atol=5e-2)
    assert driver.to_pyscf_mf().e_tot == pytest.approx(energy / Hartree, rel=1e-8)


@pytest.mark.sella
def test_short_optimization(water, uma_config, in_tmp):
    from qc_launcher.tasks import load_task

    driver = make_driver(water, uma_config)
    load_task("opt")(driver, {"max_steps": 5}, "water")
    assert (in_tmp / "water_opt.xyz").exists()


def test_inference_settings_resolution():
    from fairchem.core.units.mlip_unit.api.inference import inference_settings_default

    from qc_launcher.drivers.uma_driver import resolve_inference_settings

    assert resolve_inference_settings(None) == "default"
    assert resolve_inference_settings("batch") == "batch"
    settings = resolve_inference_settings({"internal_graph_gen_version": 3})
    assert settings.internal_graph_gen_version == 3
    assert inference_settings_default().internal_graph_gen_version == 2  # the shared preset is not mutated
    with pytest.raises(ValueError, match="Unknown inference_settings"):
        resolve_inference_settings({"no_such_field": 1})


@pytest.mark.skipif(find_spec("nvalchemiops") is None, reason="needs nvalchemi-toolkit-ops")
def test_nvalchemiops_graph_matches_default(water, uma_config):
    reference = make_driver(water, uma_config)
    nvidia = make_driver(water, uma_config, inference_settings={"internal_graph_gen_version": 3})
    assert nvidia.compute_energy() == pytest.approx(reference.compute_energy(), abs=1e-4)
    np.testing.assert_allclose(nvidia.compute_forces(), reference.compute_forces(), atol=1e-3)
