"""UMA configuration and calculator boundaries, without loading model weights."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fairchem.core")

from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings, validate_uma_atoms_data

from qc_launcher.drivers import uma_driver


@pytest.fixture
def predictor(monkeypatch):
    predictor = SimpleNamespace(
        dataset_to_tasks={"omol": [SimpleNamespace(property="energy"), SimpleNamespace(property="forces")]},
        inference_settings=deepcopy(guess_inference_settings("batch")),
        validate_atoms_data=validate_uma_atoms_data,
    )
    monkeypatch.setattr(uma_driver.pretrained_mlip, "load_predict_unit", lambda *args, **kwargs: predictor)
    return predictor


@pytest.mark.parametrize("setting", [None, "batch", {}])
def test_inference_settings_do_not_share_mutable_fields(setting):
    original = deepcopy(guess_inference_settings("batch" if setting == "batch" else "default"))
    settings = guess_inference_settings(uma_driver.resolve_inference_settings(setting))
    settings.predict_untrained_forces.add("test_task")
    try:
        assert guess_inference_settings("batch" if setting == "batch" else "default") == original
    finally:
        settings.predict_untrained_forces.discard("test_task")


@pytest.mark.parametrize(
    "setting",
    [
        False,
        1,
        ["batch"],
        "",
        "unknown",
        {"preset": "unknown"},
        {"tf32": "false"},
        {"predict_untrained_forces": "omol"},
    ],
)
def test_invalid_inference_settings_fail_clearly(setting):
    with pytest.raises(ValueError, match="inference_settings"):
        uma_driver.resolve_inference_settings(setting)


def test_yaml_task_lists_are_isolated_from_the_config():
    config = {"preset": "batch", "predict_untrained_forces": ["omol"]}
    settings = uma_driver.resolve_inference_settings(config)
    assert settings.predict_untrained_forces == {"omol"}
    settings.predict_untrained_forces.add("omat")
    assert config == {"preset": "batch", "predict_untrained_forces": ["omol"]}


def test_task_metadata_is_correct_for_single_task_materials(water, predictor):
    predictor.dataset_to_tasks = {"omat": predictor.dataset_to_tasks["omol"]}
    water.info.update(charge=1, multiplicity=2)
    driver = uma_driver.UMADriver(water, {"model_path": "unused.pt"}, verbose=False)
    original = water.info.copy()
    calc = driver.to_ase_calc()
    graph = calc.a2g(water)
    assert calc.task_name == "omat"
    assert graph.spin.item() == 0
    assert graph.charge.item() == 0
    assert water.info == original  # the physical charge and multiplicity remain available to other drivers


def test_molecular_graph_uses_multiplicity_and_validates_before_hessian(water, predictor):
    water.info.update(charge=1, multiplicity=2)
    driver = uma_driver.UMADriver(water, {"model_path": "unused.pt"}, verbose=False)
    graph = driver.calc.a2g(water)
    assert graph.spin.item() == 2
    assert graph.charge.item() == 1
    water.info["charge"] = 0.5
    with pytest.raises(TypeError, match="charge"):
        driver.compute_hessian()


def test_null_atomic_references_disable_sidecar(in_tmp, water, monkeypatch, predictor):
    (in_tmp / "iso_atom_elem_refs.yaml").write_text("omol_elem_refs: {1: {0: -1.0}}")
    captured = {}

    def load(*args, **kwargs):
        captured.update(kwargs)
        return predictor

    monkeypatch.setattr(uma_driver.pretrained_mlip, "load_predict_unit", load)
    driver = uma_driver.UMADriver(
        water, {"model_path": str(in_tmp / "model.pt"), "atom_refs_path": None}, verbose=False
    )
    driver.build_calc()
    assert captured["atom_refs"] is None


def test_explicit_missing_atomic_references_are_reported(in_tmp, water, predictor):
    driver = uma_driver.UMADriver(
        water, {"model_path": "unused.pt", "atom_refs_path": str(in_tmp / "missing.yaml")}, verbose=False
    )
    with pytest.raises(FileNotFoundError):
        driver.build_calc()


@pytest.mark.parametrize(
    "options",
    [{"batch_size": 0}, {"batch_size": -1}, {"batch_size": 1.5}, {"finite_diff_step": 0}, {"finite_diff_step": np.nan}],
)
def test_invalid_hessian_settings_fail_before_prediction(water, predictor, options):
    driver = uma_driver.UMADriver(water, {"model_path": "unused.pt", **options}, verbose=False)
    with pytest.raises(ValueError, match="batch_size|finite_diff_step"):
        driver.compute_hessian()
