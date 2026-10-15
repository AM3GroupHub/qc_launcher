import argparse

import pytest

from qc_launcher import cli
from qc_launcher.utils.config import ConfigDeprecationWarning

from .conftest import write_xyz, write_yaml
from .helpers import FakeDriver


def test_parser_has_all_subcommands():
    parser = cli.build_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert set(actions[0].choices) == {"qc", "md", "gsm", "neb", "autoneb", "pysis"}


def test_normalize_task_config():
    assert cli.normalize_task_config({"opt": True}, "opt") == {}
    assert cli.normalize_task_config({}, "opt") is None
    with pytest.warns(ConfigDeprecationWarning):
        assert cli.normalize_task_config({"md": {"dtamp": 50}}, "md") == {"tdamp": 50}
    with pytest.raises(ValueError, match="Expected a mapping or `true`"):
        cli.normalize_task_config({"opt": 3}, "opt")
    with pytest.raises(ValueError, match="requires a `md`"):
        cli.require_task_config({}, "md")


@pytest.fixture
def fake_driver_factory(monkeypatch):
    created = {}

    def factory(block, atoms, *, verbose=True):
        created["driver"] = FakeDriver(atoms, block, verbose)
        return created["driver"]

    monkeypatch.setattr(cli, "driver_from_config", factory)
    return created


def test_qc_single_point_and_forces(in_tmp, water, fake_driver_factory, capsys):
    write_xyz(in_tmp / "water.xyz", water)
    write_yaml(
        in_tmp / "run.yaml",
        {
            "inputfile": "water.xyz",
            "charge": 0,
            "multiplicity": 1,
            "driver": {"name": "fake", "k": 2.0},
            "forces": True,
        },
    )
    cli.main(["qc", "run.yaml"])
    driver = fake_driver_factory["driver"]
    assert driver.energy_calls == 1
    assert driver.atoms.info == {"charge": 0, "multiplicity": 1}
    out = capsys.readouterr().out
    assert "Total Energy        [eV]" in out and "Forces [eV/Angstrom]" in out


def test_forces_flag_must_be_true(in_tmp, water, fake_driver_factory):
    write_xyz(in_tmp / "water.xyz", water)
    write_yaml(in_tmp / "run.yaml", {"inputfile": "water.xyz", "driver": {"name": "fake"}, "forces": "yes"})
    with pytest.raises(ValueError, match="`forces` must be set to `true`"):
        cli.main(["qc", "run.yaml"])


def test_md_requires_block(in_tmp, water, fake_driver_factory):
    write_xyz(in_tmp / "water.xyz", water)
    write_yaml(in_tmp / "run.yaml", {"inputfile": "water.xyz", "driver": {"name": "fake"}})
    with pytest.raises(ValueError, match="requires a `md` configuration block"):
        cli.main(["md", "run.yaml"])


def test_multi_structure_loading(in_tmp, water):
    import ase.io

    ase.io.write(in_tmp / "path.xyz", [water, water, water])
    structures, stem = cli.load_structures({"inputfile": "path.xyz", "charge": -1}, multi=True)
    assert stem == "path" and len(structures) == 3
    assert all(frame.info["charge"] == -1 for frame in structures)
    single, _ = cli.load_structures({"inputfile": "path.xyz"})
    assert len(single) == 3  # last frame of the file


def test_extxyz_charge_and_multiplicity_become_plain_ints(in_tmp):
    (in_tmp / "fe.xyz").write_text("1\ncharge=2 multiplicity=5\nFe 0 0 0\n")
    atoms, _ = cli.load_structures({"inputfile": "fe.xyz"})
    assert type(atoms.info["charge"]) is int and atoms.info["charge"] == 2
    assert type(atoms.info["multiplicity"]) is int and atoms.info["multiplicity"] == 5
