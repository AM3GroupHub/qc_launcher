import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from qc_launcher.tasks.common import check_same_species, sella_hessian_function, timed, write_path_results

from .helpers import FakeDriver


def test_timed_prints_even_on_error(capsys):
    with timed("Thing"):
        pass
    with pytest.raises(RuntimeError):
        with timed("Broken"):
            raise RuntimeError("boom")
    out = capsys.readouterr().out
    assert "Thing completed in" in out and "Broken completed in" in out


def test_check_same_species(water):
    check_same_species([water, water.copy()])
    other = water.copy()
    other.set_chemical_symbols(["O", "H", "F"])
    with pytest.raises(ValueError, match="same atoms"):
        check_same_species([water, other])


def test_write_path_results(in_tmp, water):
    images = []
    for energy in (0.0, 1.0, 0.5):
        image = water.copy()
        image.calc = SinglePointCalculator(image, energy=energy, forces=np.zeros((3, 3)))
        images.append(image)
    assert write_path_results(images, "path") == 1
    assert (in_tmp / "path_neb.xyz").exists() and (in_tmp / "path_ts.xyz").exists()


def test_sella_hessian_function(water):
    driver = FakeDriver(water, {"k": 2.0})
    hessian = sella_hessian_function(driver)(water)
    np.testing.assert_allclose(hessian, 2.0 * np.eye(9))
