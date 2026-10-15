import pytest

from qc_launcher.utils.config import ConfigDeprecationWarning, rename_deprecated_keys


def test_deprecated_key_is_renamed_with_warning():
    with pytest.warns(ConfigDeprecationWarning, match="'dtamp' is deprecated"):
        result = rename_deprecated_keys({"dtamp": 50.0, "nsteps": 10}, "md")
    assert result == {"tdamp": 50.0, "nsteps": 10}


def test_canonical_key_wins_over_deprecated():
    with pytest.warns(ConfigDeprecationWarning, match="using 'threads'"):
        result = rename_deprecated_keys({"thread": 2, "threads": 8}, "driver")
    assert result == {"threads": 8}


def test_untouched_config_is_copied():
    original = {"xc": "PBE", "basis": "sto-3g"}
    result = rename_deprecated_keys(original, "driver")
    assert result == original and result is not original


def test_unknown_section_and_none():
    assert rename_deprecated_keys({"a": 1}, "nope") == {"a": 1}
    assert rename_deprecated_keys(None, "md") == {}
