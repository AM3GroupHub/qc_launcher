import pytest

pytestmark = pytest.mark.pygsm


def test_resolve_ts_search_and_legacy_climb():
    from qc_launcher.tasks.gsm import resolve_ts_search
    from qc_launcher.utils.config import ConfigDeprecationWarning

    assert resolve_ts_search({}) == "climb"
    assert resolve_ts_search({"ts_search": "exact"}) == "exact"
    with pytest.warns(ConfigDeprecationWarning, match="climb"):
        assert resolve_ts_search({"climb": False}) == "exact"
    with pytest.raises(ValueError, match="ts_search"):
        resolve_ts_search({"ts_search": "sideways"})


def test_run_gsm_input_validation(water):
    from qc_launcher.tasks.gsm import run_gsm

    from .helpers import FakeDriver

    driver = FakeDriver(water)
    with pytest.raises(NotImplementedError):
        run_gsm(driver, {"gsm_type": "SE_GSM"}, [water])
    with pytest.raises(ValueError, match="exactly two"):
        run_gsm(driver, {}, [water])
    with pytest.raises(ValueError, match="optimizer_method"):
        run_gsm(driver, {"optimizer_method": "newton"}, [water, water.copy()])
