"""ByteQC backend against the pyscf CPU backend; needs QC_LAUNCHER_BYTEQC_PYTHON (GPU environment)."""

import pytest

from qc_launcher.tasks.refine import run_refine

from ..conftest import byteqc_python
from ..helpers import FakeDriver

pytestmark = pytest.mark.byteqc


@pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd(t)"])
def test_byteqc_matches_pyscf_backend(in_tmp, water, method):
    frozen = method != "mp2"  # ByteQC DF-MP2 has no frozen core
    base = {"method": method, "basis": "cc-pVDZ", "frozen_core": frozen, "verbose": 0}
    cpu = run_refine(FakeDriver(water), {**base, "backend": "pyscf", "python": byteqc_python()}, "water_cpu")
    gpu = run_refine(FakeDriver(water), {**base, "backend": "byteqc", "python": byteqc_python()}, "water_gpu")
    assert gpu["e_hf"] == pytest.approx(cpu["e_hf"], abs=1e-8)
    assert gpu["e_corr"] == pytest.approx(cpu["e_corr"], abs=1e-6)
