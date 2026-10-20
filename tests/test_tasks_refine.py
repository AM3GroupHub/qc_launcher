"""Refinement task plumbing without a quantum-chemistry backend: a stub runner stands in for the real one."""

import json
import sys
import textwrap

import pytest

from qc_launcher.tasks import refine
from qc_launcher.tasks.refine import composite_thermochemistry, run_refine

from .helpers import FakeDriver

THERMO = {
    "E0": (-76.0, "Eh"),
    "E_0K": (-75.98, "Eh"),
    "H_tot": (-75.97, "Eh"),
    "G_tot": (-75.99, "Eh"),
    "G_tot_qrrho": (-75.991, "Eh"),
}


def test_composite_replaces_e0():
    composite = composite_thermochemistry(THERMO, -76.3)
    assert composite["E_0K_refined"] == pytest.approx(-76.28)
    assert composite["G_tot_qrrho_refined"] == pytest.approx(-76.291)
    assert "H_tot_qrrho_refined" not in composite
    json_layout = {k: {"value": v[0], "unit": v[1]} for k, v in THERMO.items()}
    assert composite_thermochemistry(json_layout, -76.3)["G_tot_refined"] == pytest.approx(-76.29)
    assert composite_thermochemistry({}, -76.3) == {}


@pytest.fixture
def stub_runner(tmp_path, monkeypatch):
    """A runner that echoes the input spec and returns fixed energies (fails if the spec asks for 'fail')."""
    script = tmp_path / "stub_runner.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            spec = json.load(open(sys.argv[1]))
            if spec["basis"] == "fail":
                json.dump({"error": "ValueError: boom", "traceback": "stub traceback"}, open(sys.argv[2], "w"))
                sys.exit(1)
            out = {"backend": spec["backend"], "method": spec["method"], "e_hf": -76.0, "e_corr": -0.3,
                   "e_total": -76.3, "e_corr_ccsd": -0.29, "e_t": -0.01, "warnings": ["stub warning"],
                   "timings": {}, "versions": {}, "spec_echo": spec}
            json.dump(out, open(sys.argv[2], "w"))
            """
        )
    )
    monkeypatch.setattr(refine, "runner_path", lambda: str(script))
    monkeypatch.setattr(refine, "probe_backend", lambda python, backend, **kwargs: None)
    return script


def test_run_refine_writes_outputs_and_composite(in_tmp, water, stub_runner, capsys):
    driver = FakeDriver(water)
    result = run_refine(driver, {"backend": "pyscf", "method": "ccsd(t)", "basis": "cc-pVDZ"}, "water", thermo=THERMO)
    assert result["e_total"] == -76.3
    assert result["composite"]["G_tot_refined"] == pytest.approx(-76.29)
    assert result["spec_echo"]["symbols"] == ["O", "H", "H"] and result["spec_echo"]["charge"] == 0
    saved = json.loads((in_tmp / "water_refine.json").read_text())
    assert saved["composite"]["E_0K_refined"] == pytest.approx(-76.28)
    assert (in_tmp / "water_refine_input.json").exists() and (in_tmp / "water_refine.log").exists()
    out = capsys.readouterr().out
    assert "Warning: stub warning" in out and "Refined E_total" in out and "G_tot_refined" in out


def test_run_refine_reads_thermo_json_from_disk(in_tmp, water, stub_runner):
    (in_tmp / "water_thermo.json").write_text(json.dumps({k: {"value": v[0], "unit": v[1]} for k, v in THERMO.items()}))
    result = run_refine(FakeDriver(water), {"method": "mp2"}, "water")
    assert result["composite"]["H_tot_refined"] == pytest.approx(-76.27)


def test_run_refine_without_thermo(in_tmp, water, stub_runner):
    result = run_refine(FakeDriver(water), {"method": "ccsd"}, "water")
    assert result["composite"] == {}


def test_runner_failure_is_reported(in_tmp, water, stub_runner):
    with pytest.raises(RuntimeError, match="boom"):
        run_refine(FakeDriver(water), {"basis": "fail"}, "water")


def test_pythonpath_is_used_by_probe_and_runner(in_tmp, water, monkeypatch):
    package = in_tmp / "external" / "byteqc"
    package.mkdir(parents=True)
    for name in ("__init__", "cucc", "cump2"):
        (package / f"{name}.py").write_text("")
    script = in_tmp / "runner.py"
    script.write_text(
        "import byteqc.cucc, byteqc.cump2, json, sys\n"
        "json.dump({'e_hf': -1.0, 'e_corr': -0.1, 'e_total': -1.1}, open(sys.argv[2], 'w'))\n"
    )
    monkeypatch.setattr(refine, "runner_path", lambda: str(script))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = run_refine(FakeDriver(water), {"backend": "byteqc", "pythonpath": str(package.parent)}, "water")
    assert result["e_total"] == -1.1


def test_nonzero_runner_exit_rejects_written_result(in_tmp, water, stub_runner):
    with stub_runner.open("a") as handle:
        handle.write("sys.exit(7)\n")
    with pytest.raises(RuntimeError, match="exit code 7"):
        run_refine(FakeDriver(water), {}, "water")
    assert not (in_tmp / "water_refine.json").exists()


def test_validation(in_tmp, water, stub_runner):
    radical = water.copy()
    radical.info["multiplicity"] = 2
    with pytest.raises(ValueError, match="closed-shell"):
        run_refine(FakeDriver(radical), {}, "water")
    with pytest.raises(ValueError, match="refine.method"):
        run_refine(FakeDriver(water), {"method": "cisd"}, "water")
    with pytest.raises(ValueError, match="refine.backend"):
        run_refine(FakeDriver(water), {"backend": "orca"}, "water")
    with pytest.raises(ValueError, match="does not exist"):
        run_refine(FakeDriver(water), {"python": "/no/such/python"}, "water")


def test_cli_runs_refine_after_freq(in_tmp, water, stub_runner, monkeypatch):
    from qc_launcher import cli

    from .conftest import write_xyz, write_yaml

    calls = []

    def fake_freq(driver, config, filename):
        calls.append("freq")
        return THERMO

    original = refine.run_refine

    def spy_refine(driver, config, filename, thermo=None):
        calls.append(("refine", thermo is THERMO))
        return original(driver, config, filename, thermo=thermo)

    monkeypatch.setattr(
        cli, "driver_from_config", lambda block, atoms, *, verbose=True: FakeDriver(atoms, block, verbose)
    )
    monkeypatch.setattr("qc_launcher.tasks.freq.run_freq", fake_freq)
    monkeypatch.setattr(refine, "run_refine", spy_refine)
    write_xyz(in_tmp / "water.xyz", water)
    write_yaml(
        in_tmp / "run.yaml",
        {
            "inputfile": "water.xyz",
            "driver": {"name": "fake"},
            "freq": True,
            "refine": {"method": "mp2", "python": sys.executable},
        },
    )
    cli.main(["qc", "run.yaml"])
    assert calls == ["freq", ("refine", True)]
    assert (in_tmp / "water_refine.json").exists()
