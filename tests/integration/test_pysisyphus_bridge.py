import numpy as np
import pytest
from ase.units import Bohr, Hartree

pytestmark = [pytest.mark.pysisyphus, pytest.mark.tblite]


def test_bridge_matches_driver(in_tmp, water):  # pysisyphus creates qm_calcs/ in the working directory
    from qc_launcher.drivers.pysisyphus_calc import QCSisyphusCalc
    from qc_launcher.drivers.tblite_driver import TBLiteDriver

    calc = QCSisyphusCalc(driver={"name": "tblite", "method": "GFN2-xTB"}, charge=0, mult=1)
    symbols = water.get_chemical_symbols()
    coords = water.get_positions().ravel() / Bohr
    driver = TBLiteDriver(water, {"method": "GFN2-xTB"}, verbose=False)

    energy = calc.get_energy(symbols, coords)["energy"]
    assert energy == pytest.approx(driver.compute_energy() / Hartree, abs=1e-9)
    forces = calc.get_forces(symbols, coords)["forces"]
    np.testing.assert_allclose(forces.reshape(-1, 3), driver.compute_forces() * Bohr / Hartree, atol=1e-8)
    h1 = calc.get_hessian(symbols, coords)["hessian"]
    h2 = calc.get_hessian(symbols, coords)["hessian"]
    np.testing.assert_allclose(h1, h2)  # the cached Hessian is not rescaled on repeated calls
    np.testing.assert_allclose(h1, driver.compute_hessian(hess_format="ase") * Bohr**2 / Hartree, atol=1e-8)
