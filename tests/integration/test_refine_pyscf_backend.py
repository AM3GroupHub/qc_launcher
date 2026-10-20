"""The real runner with the pyscf (CPU) backend against direct pyscf calls; water / STO-3G, seconds."""

import pytest

from qc_launcher.tasks.refine import run_refine

from ..helpers import FakeDriver

pytestmark = pytest.mark.pyscf


def reference(water, method, frozen):
    from pyscf import cc, gto, scf
    from pyscf.mp.dfmp2_native import DFMP2

    mol = gto.M(
        atom=[(s, p) for s, p in zip(water.get_chemical_symbols(), water.get_positions())], basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()
    if method == "mp2":
        return e_hf, DFMP2(mf, frozen=frozen).kernel()
    mcc = cc.CCSD(mf, frozen=frozen)
    mcc.conv_tol = 1e-8
    e_ccsd = mcc.kernel()[0]
    return e_hf, e_ccsd + (mcc.ccsd_t() if method == "ccsd(t)" else 0.0)


@pytest.mark.parametrize("method", ["mp2", "ccsd", "ccsd(t)"])
def test_pyscf_backend_matches_direct_pyscf(in_tmp, water, method):
    config = {"backend": "pyscf", "method": method, "basis": "sto-3g", "frozen_core": True, "verbose": 0}
    result = run_refine(FakeDriver(water), config, "water")
    e_hf, e_corr = reference(water, method, frozen=1)
    assert result["n_frozen"] == 1
    assert result["e_hf"] == pytest.approx(e_hf, rel=0, abs=1e-9)
    assert result["e_corr"] == pytest.approx(e_corr, rel=0, abs=1e-8)
    assert result["e_total"] == pytest.approx(e_hf + e_corr, rel=0, abs=1e-8)
    assert result["versions"]["pyscf"]


def test_all_electron_option(in_tmp, water):
    result = run_refine(
        FakeDriver(water),
        {"backend": "pyscf", "method": "mp2", "basis": "sto-3g", "frozen_core": False, "verbose": 0},
        "water",
    )
    e_hf, e_corr = reference(water, "mp2", frozen=None)
    assert result["n_frozen"] == 0
    assert result["e_corr"] == pytest.approx(e_corr, rel=0, abs=1e-8)


@pytest.mark.parametrize("density_fit", [True, False])
def test_ccsd_uses_correlation_auxbasis(in_tmp, water, density_fit):
    from pyscf import df, gto, scf
    from pyscf.cc.dfccsd import RCCSD

    mol = gto.M(atom=list(zip(water.get_chemical_symbols(), water.get_positions())), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit(auxbasis="weigend")
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()
    mcc = RCCSD(mf, frozen=1)
    mcc.with_df = df.DF(mol, auxbasis="cc-pvdz-ri")
    mcc.conv_tol = 1e-8
    e_corr = mcc.kernel()[0]
    result = run_refine(
        FakeDriver(water),
        {
            "method": "ccsd",
            "basis": "sto-3g",
            "auxbasis": "cc-pvdz-ri",
            "scf": {"density_fit": density_fit, "auxbasis": "weigend"},
            "verbose": 0,
        },
        "water",
    )
    assert result["e_hf"] == pytest.approx(e_hf, rel=0, abs=1e-9)
    assert result["e_corr"] == pytest.approx(e_corr, rel=0, abs=1e-8)
