"""cuEST accelerator against native gpu4pyscf on the same GPU (water, PBE0/def2-SVP, density fitting)."""

import numpy as np
import pytest
from ase.units import Bohr, Hartree

pytestmark = [pytest.mark.cuest, pytest.mark.gpu]

BASE = {
    "xc": "PBE0",
    "basis": "def2-SVP",
    "with_df": True,
    "auxbasis": "def2-universal-jkfit",
    "with_gpu": True,
    "verbosity": 0,
    "threads": 2,
    "scf_conv_tol": 1e-10,
}
CUEST = {**BASE, "accelerator": "cuest", "cuest": {"match_native_grids": True}}


def make(atoms, config, **overrides):
    from qc_launcher.drivers.pyscf_driver import PySCFDriver

    cfg = {**config, **overrides}
    return PySCFDriver(atoms, cfg, verbose=False)


@pytest.fixture(scope="module")
def native_reference(request):
    """Native gpu4pyscf with the grid settings cuEST uses (Stratmann, no radii adjustment)."""
    from ase.build import molecule
    from pyscf.dft.gen_grid import stratmann

    water = molecule("H2O")
    water.info.update(charge=0, multiplicity=1)
    driver = make(water, BASE)
    for grids in (driver.method.grids, driver.method.nlcgrids):
        grids.becke_scheme = stratmann
        grids.radii_adjust = None
    driver.method.reset(driver.method.mol)
    energy = driver.compute_energy()
    # cuEST gradients always include the grid-weight response; the native default leaves it out
    grad = driver.method.nuc_grad_method()
    grad.grid_response = True
    forces = -np.asarray(grad.kernel()) * Hartree / Bohr
    hessian = driver.compute_hessian(hess_format="ase")  # analytic, grids_response=True
    # finite differences of the native gradients with grid response: what the cuEST path must reproduce
    from qc_launcher.drivers.base_driver import convert_hessian
    from qc_launcher.drivers.pyscf_driver import _asnumpy, _central_difference_hessian

    mf = driver.method
    mol0 = mf.mol
    dm0 = mf.make_rdm1()

    def gradient_at(coords):
        mf.reset(mol0.set_geom_(coords, unit="Bohr", inplace=False))
        mf.kernel(dm0=dm0)
        g = mf.nuc_grad_method()
        g.grid_response = True
        return _asnumpy(g.kernel())

    fd_hessian = convert_hessian(_central_difference_hessian(mol0, gradient_at, 5e-3 / Bohr), mol0.natm, "ase")
    mf.reset(mol0)  # leave the reference method at the undisplaced geometry
    return water, energy, forces, hessian, fd_hessian


def test_energy_and_forces_match_native(native_reference):
    water, e_ref, f_ref, _, _ = native_reference
    driver = make(water.copy(), CUEST)
    assert driver.accelerator == "cuest"
    assert driver.compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)
    np.testing.assert_allclose(driver.compute_forces(), f_ref, atol=1e-5)


def test_default_math_mode_and_disable(native_reference):
    water, e_ref, _, _, _ = native_reference
    emulated = make(water.copy(), CUEST, cuest={"match_native_grids": True, "math_mode": "default"})
    assert emulated.method.math_mode == "default"
    assert emulated.compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)
    partial = make(water.copy(), CUEST, cuest={"match_native_grids": True, "disable": ["xc"]})
    assert partial.method.turn_on_cuest_xc is False
    assert partial.compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)


def test_hessians(native_reference):
    water, _, _, h_analytic, h_fd_native = native_reference
    fd = make(water.copy(), CUEST).compute_hessian(hess_format="ase")  # finite differences of cuEST gradients
    np.testing.assert_allclose(fd, fd.T, atol=1e-8)
    np.testing.assert_allclose(fd, h_fd_native, atol=1e-4)
    # gpu4pyscf's analytic Hessian treats the grid response differently from its gradients (~0.1% here)
    np.testing.assert_allclose(fd, h_analytic, atol=0.2)
    # native analytic Hessian from the shadow method seeded with the cuEST density (SCF-noise level agreement)
    analytic = make(water.copy(), CUEST, numerical_hessian=False).compute_hessian(hess_format="ase")
    np.testing.assert_allclose(analytic, h_analytic, atol=1e-4)


def test_scf_follows_in_place_moves(native_reference):
    water, e_ref, _, _, _ = native_reference
    atoms = water.copy()
    driver = make(atoms, CUEST)
    e1 = driver.compute_energy()
    atoms.positions[0, 0] += 0.2
    e2 = driver.compute_energy()
    assert abs(e2 - e1) > 1e-3
    atoms.positions[0, 0] -= 0.2
    assert driver.compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)


def test_cpcm_and_uks_parity(water):
    from ase import Atoms

    cpcm = {"solvation": {"model": "C-PCM", "epsilon": 78.3553}}
    ref = make(water.copy(), BASE, **cpcm)
    from pyscf.dft.gen_grid import stratmann

    ref.method.grids.becke_scheme = stratmann
    ref.method.grids.radii_adjust = None
    ref.method.with_solvent.surface_discretization_method = "ISWIG"
    ref.method.reset(ref.method.mol)
    e_ref = ref.compute_energy()
    assert make(water.copy(), CUEST, **cpcm).compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)

    radical = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])
    radical.info.update(charge=0, multiplicity=2)
    ref = make(radical.copy(), BASE)
    ref.method.grids.becke_scheme = stratmann
    ref.method.grids.radii_adjust = None
    ref.method.reset(ref.method.mol)
    e_ref = ref.compute_energy()
    assert make(radical.copy(), CUEST).compute_energy() == pytest.approx(e_ref, abs=1e-6 * Hartree)


@pytest.mark.sella
def test_opt_freq_and_md_through_tasks(water, in_tmp):
    from qc_launcher.tasks import load_task

    driver = make(water, CUEST)
    load_task("opt")(driver, {"max_steps": 30}, "water")
    np.testing.assert_allclose(driver.compute_forces(), 0.0, atol=2e-2)
    load_task("freq")(driver, {"qrrho": True}, "water")
    import json

    frequencies = json.loads((in_tmp / "water_thermo.json").read_text())["frequencies_cm1"]
    assert len(frequencies) == 3 and min(frequencies) > 0
    load_task("md")(
        driver,
        {"ensemble": "NVE", "minimize": False, "timestep": 0.5, "nsteps": 5, "loginterval": 5, "logfile": "md.log"},
        "water",
    )
    assert (in_tmp / "water_md.traj").exists()
