from types import SimpleNamespace

import pytest

from qc_launcher.drivers import driver_factory
from qc_launcher.drivers.driver_factory import driver_from_config, get_driver
from qc_launcher.utils.config import ConfigDeprecationWarning

from .helpers import FakeDriver


def test_unknown_driver_lists_available_names():
    with pytest.raises(ValueError, match="mace, pyscf, tblite, uma"):
        get_driver("nope", None, {})


def test_missing_optional_package_gives_install_hint(monkeypatch):
    def boom(name, package=None):
        raise ModuleNotFoundError("No module named 'fairchem'", name="fairchem")

    monkeypatch.setattr(driver_factory, "import_module", boom)
    with pytest.raises(
        ModuleNotFoundError, match=r"driver 'uma' requires optional dependency 'fairchem'.*\[fairchem\]"
    ):
        get_driver("UMA", None, {})


def test_driver_from_config_requires_name():
    with pytest.raises(ValueError, match="`name`"):
        driver_from_config({"xc": "PBE"}, None)


def test_driver_block_deprecated_key_is_renamed(monkeypatch, water):
    monkeypatch.setitem(driver_factory._DRIVER_MAP, "fake", ("fake_module", "FakeDriver", None))
    monkeypatch.setattr(
        driver_factory, "import_module", lambda name, package=None: SimpleNamespace(FakeDriver=FakeDriver)
    )
    with pytest.warns(ConfigDeprecationWarning):
        driver = driver_from_config({"name": "Fake", "thread": 4}, water, verbose=False)
    assert isinstance(driver, FakeDriver)
    assert driver.config == {"threads": 4}
    assert driver.atoms is water
