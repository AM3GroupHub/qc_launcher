from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

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
    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings

    from qc_launcher.drivers.uma_driver import resolve_inference_settings

    original = deepcopy(guess_inference_settings("default"))
    assert resolve_inference_settings(None) == original
    assert resolve_inference_settings("batch") == guess_inference_settings("batch")
    settings = resolve_inference_settings({"internal_graph_gen_version": 3})
    assert settings.internal_graph_gen_version == 3
    assert guess_inference_settings("default") == original
    with pytest.raises(ValueError, match="Unknown inference_settings"):
        resolve_inference_settings({"no_such_field": 1})


@pytest.mark.skipif(find_spec("nvalchemiops") is None, reason="needs nvalchemi-toolkit-ops")
def test_nvalchemiops_graph_matches_default(water, uma_config):
    reference = make_driver(water, uma_config)
    nvidia = make_driver(water, uma_config, inference_settings={"internal_graph_gen_version": 3})
    assert nvidia.compute_energy() == pytest.approx(reference.compute_energy(), rel=0, abs=1e-4)
    np.testing.assert_allclose(nvidia.compute_forces(), reference.compute_forces(), atol=1e-3)


def test_default_task_name_handles_single_task_checkpoints():
    from types import SimpleNamespace

    from qc_launcher.drivers.uma_driver import default_task_name

    multi = SimpleNamespace(dataset_to_tasks={"omol": [], "omat": [], "oc20": []})
    single = SimpleNamespace(dataset_to_tasks={"omat": []})
    assert default_task_name(multi, {}) == "omol"
    assert default_task_name(single, {}) == "omat"
    assert default_task_name(single, {"task_name": "omol"}) == "omol"  # explicit values are passed through


@pytest.mark.parametrize("variant,task", [("d4rev", "omol"), ("r2scan", "omat")])
def test_finetuned_checkpoint_matches_direct_fairchem(water, uma_model_path, variant, task):
    from ase.build import bulk
    from fairchem.core import FAIRChemCalculator

    path = Path(uma_model_path).with_name(f"uma-s-1p2p1-ft-{variant}.pt")
    if not path.is_file():
        pytest.skip(f"Fine-tuned checkpoint not found: {path}")
    atoms = water.copy() if task == "omol" else bulk("Si", "diamond", a=5.43)
    atoms.info.update(charge=0, multiplicity=1)
    driver = make_driver(atoms, {"model_path": str(path), "inference_settings": "batch"})
    # Exercise Hessians before an ASE energy call has validated/defaulted the input metadata.
    hessian = driver.compute_hessian(hess_format="ase")
    assert driver.calc.task_name == task
    assert np.isfinite(hessian).all()
    reference = atoms.copy()
    reference.info.update(charge=0, spin=1 if task == "omol" else 0)
    reference.calc = FAIRChemCalculator(driver.calc.predictor, task_name=task)
    assert driver.compute_energy() == pytest.approx(reference.get_potential_energy(), rel=0, abs=1e-6)
    np.testing.assert_allclose(driver.compute_forces(), reference.get_forces(), rtol=0, atol=1e-6)
    eps = 5e-3
    reference.positions[0, 0] += eps
    plus = reference.get_forces().copy()
    reference.positions[0, 0] -= 2 * eps
    minus = reference.get_forces().copy()
    np.testing.assert_allclose(hessian[:, 0], ((minus - plus) / (2 * eps)).ravel(), rtol=0, atol=5e-2)
