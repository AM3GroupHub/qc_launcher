"""ByteQC backend against the pyscf CPU backend; needs QC_LAUNCHER_BYTEQC_PYTHON (GPU environment)."""

import json

import pytest

from qc_launcher.tasks import refine
from qc_launcher.tasks.refine import run_refine

from ..conftest import byteqc_python
from ..helpers import FakeDriver

pytestmark = pytest.mark.byteqc


@pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd(t)"])
def test_byteqc_matches_pyscf_backend(in_tmp, water, method):
    frozen = method != "mp2"  # ByteQC DF-MP2 has no frozen core
    base = {
        "method": method,
        "basis": "cc-pVDZ",
        "frozen_core": frozen,
        "gpu_memory_gb": 1,
        "cpu_memory_gb": 1,
        "timeout_s": 120,
        "verbose": 0,
    }
    cpu = run_refine(FakeDriver(water), {**base, "backend": "pyscf", "python": byteqc_python()}, "water_cpu")
    gpu = run_refine(FakeDriver(water), {**base, "backend": "byteqc", "python": byteqc_python()}, "water_gpu")
    assert gpu["e_hf"] == pytest.approx(cpu["e_hf"], rel=0, abs=1e-8)
    assert gpu["e_corr"] == pytest.approx(cpu["e_corr"], rel=0, abs=1e-6)


@pytest.mark.parametrize("gpu_scf", [False, True])
def test_frozen_ccsd_t_with_custom_auxbasis_and_multiple_blocks(in_tmp, water, monkeypatch, gpu_scf):
    base = {
        "method": "ccsd(t)",
        "basis": "cc-pVDZ",
        "auxbasis": "cc-pvdz-ri",
        "frozen_core": True,
        "scf": {"density_fit": not gpu_scf, "auxbasis": "weigend"},
        "gpu_memory_gb": 1,
        "cpu_memory_gb": 1,
        "timeout_s": 120,
        "verbose": 0,
        "python": byteqc_python(),
    }
    cpu = run_refine(FakeDriver(water), {**base, "backend": "pyscf"}, "water_cpu")
    # Force multiple DF blocks on a small molecule so the frozen-orbital buffer reuse is exercised.
    script = in_tmp / "small_blocks_runner.py"
    script.write_text(
        "import runpy\n"
        "from pyscf import df\n"
        "df.DF.blockdim = 8\n"
        f"runpy.run_path({refine.runner_path()!r}, run_name='__main__')\n"
    )
    monkeypatch.setattr(refine, "runner_path", lambda: str(script))
    gpu = run_refine(
        FakeDriver(water),
        {**base, "backend": "byteqc", "scf": {**base["scf"], "gpu": gpu_scf}},
        "water_gpu",
    )
    assert gpu["n_frozen"] == 1
    assert gpu["scf_gpu"] is gpu_scf
    for key in ("e_hf", "e_corr_ccsd", "e_t", "e_total"):
        assert gpu[key] == pytest.approx(cpu[key], rel=0, abs=1e-7)
    assert json.loads((in_tmp / "water_gpu_refine.json").read_text())["ccsd_converged"]
