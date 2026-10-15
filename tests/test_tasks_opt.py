import numpy as np
import pytest

from qc_launcher.tasks import load_task

pytestmark = [pytest.mark.sella, pytest.mark.tblite]


def make_driver(atoms):
    from qc_launcher.drivers.tblite_driver import TBLiteDriver

    return TBLiteDriver(atoms, {"verbosity": 0}, verbose=False)


def test_zero_steps_does_not_crash(in_tmp, water):
    load_task("opt")(make_driver(water), {"max_steps": 0}, "water")
    assert (in_tmp / "water_opt.xyz").exists()


def test_bond_constraint_is_kept(in_tmp, water):
    water.set_distance(0, 1, 1.2, fix=0)
    driver = make_driver(water)
    load_task("opt")(driver, {"constraints": {"fix_bond": [[0, 1]]}, "max_steps": 60}, "water")
    assert water.get_distance(0, 1) == pytest.approx(1.2, abs=1e-3)
    assert water.get_distance(0, 2) == pytest.approx(0.96, abs=0.03)  # the free bond relaxed


def test_unknown_constraint_type(water):
    from qc_launcher.tasks.opt import setup_constraints

    with pytest.raises(ValueError, match="Unknown constraint types"):
        setup_constraints(water, {"fix_everything": [1]})
    assert setup_constraints(water, {}) is None
    np.testing.assert_array_equal(water.get_positions().shape, (3, 3))
