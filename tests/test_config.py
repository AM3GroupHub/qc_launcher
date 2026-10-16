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


def test_gsm_keys_follow_pygsm_names():
    old = {
        "num_nodes": 9,
        "max_gsm_steps": 50,
        "max_opt_steps": 2,
        "ID": 3,
        "fmax": 0.05,
        "ediff": 0.1,
        "line_search": "backtrack",
    }
    with pytest.warns(ConfigDeprecationWarning):
        result = rename_deprecated_keys(old, "gsm")
    assert result == {
        "nnodes": 9,
        "max_iters": 50,
        "opt_steps": 2,
        "string_id": 3,
        "conv_gmax": 0.05,
        "conv_ediff": 0.1,
        "linesearch": "backtrack",
    }
    with pytest.warns(ConfigDeprecationWarning):
        assert rename_deprecated_keys({"max_nodes": 7}, "gsm") == {"nnodes": 7}


@pytest.mark.parametrize(
    "section, old, new",
    [
        (
            "driver",
            {"verbose": 1, "scf_max_cycle": 20, "disp": "d3bj", "numerical_hess": True, "finite_diff_eps": 1e-3},
            {
                "verbosity": 1,
                "scf_max_iter": 20,
                "dispersion": "d3bj",
                "numerical_hessian": True,
                "finite_diff_step": 1e-3,
            },
        ),
        (
            "driver",
            {"max_iter": 30, "etemp": 500.0, "solvent": {"method": "smd"}},
            {"scf_max_iter": 30, "electronic_temperature": 500.0, "solvation": {"method": "smd"}},
        ),
        ("driver.solvation", {"method": "cpcm", "eps": 4.0}, {"model": "cpcm", "epsilon": 4.0}),
        ("freq", {"Bav": "auto"}, {"bav": "auto"}),
        (
            "irc",
            {"irc_trajectory": "a.traj", "irc_output": "a.xyz", "steps": 5},
            {"trajectory": "a.traj", "outputfile": "a.xyz", "max_steps": 5},
        ),
        ("neb", {"max_neb_steps": 10, "max_opt_steps": 2}, {"max_steps": 10, "endpoint_max_steps": 2}),
        ("autoneb", {"max_neb_steps": 10, "max_opt_steps": 2}, {"max_steps": 10, "endpoint_max_steps": 2}),
        ("gsm", {"endpoint_opt_steps": 20}, {"endpoint_max_steps": 20}),
    ],
)
def test_unified_key_aliases(section, old, new):
    with pytest.warns(ConfigDeprecationWarning):
        assert rename_deprecated_keys(old, section) == new
