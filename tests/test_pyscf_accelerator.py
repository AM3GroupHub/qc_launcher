"""Validation of the cuEST accelerator settings; runs without a GPU or cuEST (errors are raised before any import)."""

import pytest

pytestmark = pytest.mark.pyscf

BASE = {"xc": "PBE0", "basis": "sto-3g", "with_df": True, "with_gpu": True, "accelerator": "cuest", "verbosity": 0}


def make(config, water):
    from qc_launcher.drivers.pyscf_driver import PySCFDriver

    return PySCFDriver(water, config, verbose=False)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"with_df": False}, "requires density fitting"),
        ({"with_gpu": False}, "requires with_gpu"),
        ({"retry_soscf": True}, "retry_soscf"),
        ({"solvation": {"model": "IEF-PCM", "solvent": "water"}}, "only C-PCM and COSMO"),
        ({"solvation": {"model": "smd", "solvent": "water"}}, "only C-PCM and COSMO"),
        ({"with_solvent": True}, "only C-PCM and COSMO"),
        ({"xc": "B97-3c"}, "3c"),
        ({"cuest": {"disable": ["grids"]}}, "unknown parts"),
        ({"cuest": {"math_mode": "fp32"}}, "math_mode"),
        ({"accelerator": "jqc"}, "Unknown accelerator"),
    ],
)
def test_unsupported_settings_are_rejected_early(water, overrides, message):
    config = {**BASE, **overrides}
    with pytest.raises(ValueError, match=message):
        make(config, water)


def test_accelerator_none_is_default(water):
    from qc_launcher.drivers.pyscf_driver import PySCFDriver

    driver = PySCFDriver(
        water, {"xc": "PBE", "basis": "sto-3g", "with_df": False, "with_gpu": False, "verbosity": 0}, verbose=False
    )
    assert driver.accelerator == "none"
    assert driver.config.get("numerical_hessian") is None  # analytic Hessian stays the default without cuEST
