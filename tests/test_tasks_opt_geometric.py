from pathlib import Path

import numpy as np
import pytest

from qc_launcher.tasks import load_task
from qc_launcher.tasks.opt_geometric import convergence_kwargs, geometric_constraints

from .helpers import FakeDriver

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_constraint_translation():
    text = geometric_constraints(
        {"fix_translation": [0], "fix_bond": [[0, 1]], "fix_angle": [[1, 0, 2]], "fix_dihedral": [[0, 1, 2, 3]]}
    )
    assert text.splitlines() == ["$freeze", "xyz 1", "distance 1 2", "angle 2 1 3", "dihedral 1 2 3 4"]
    assert geometric_constraints({}) is None
    assert geometric_constraints({"fix_bond": []}) is None
    with pytest.raises(ValueError, match="Unknown constraint types"):
        geometric_constraints({"fix_everything": [1]})


def test_convergence_kwargs_match_gaussian_defaults():
    kwargs = convergence_kwargs({})
    assert kwargs["convergence_energy"] == pytest.approx(1e-6)
    assert kwargs["convergence_gmax"] == pytest.approx(4.5e-4)
    assert kwargs["convergence_grms"] == pytest.approx(3e-4)
    assert kwargs["convergence_dmax"] == pytest.approx(1.8e-3)
    assert kwargs["convergence_drms"] == pytest.approx(1.2e-3)
    custom = convergence_kwargs({"convergence_set": "gau_tight", "fmax": 0.023})
    assert custom["convergence_set"] == "gau_tight"
    assert custom["convergence_gmax"] == pytest.approx(0.023 * 0.529177 / 27.2114, rel=1e-4)
    assert "convergence_energy" not in custom


def test_unknown_optimizer(water):
    with pytest.raises(ValueError, match="Unknown optimizer"):
        load_task("opt")(FakeDriver(water), {"optimizer": "orca"}, "water")


def test_dispatch_to_geometric(water, monkeypatch):
    from qc_launcher.tasks import opt_geometric

    calls = []
    monkeypatch.setattr(opt_geometric, "run_opt_geometric", lambda driver, config, filename: calls.append(filename))
    load_task("opt")(FakeDriver(water), {"optimizer": "GeomeTRIC"}, "water")
    assert calls == ["water"]


@pytest.mark.geometric
@pytest.mark.tblite
class TestGeometricWithTBLite:
    @staticmethod
    def make_driver(atoms):
        from qc_launcher.drivers.tblite_driver import TBLiteDriver

        return TBLiteDriver(atoms, {"verbosity": 0}, verbose=False)

    def test_minimization(self, in_tmp, water):
        from qc_launcher.tasks.opt_geometric import run_opt_geometric

        water.set_distance(0, 1, 1.1, fix=0)
        driver = self.make_driver(water)
        assert run_opt_geometric(driver, {"max_steps": 60}, "water") is True
        assert water.get_distance(0, 1) == pytest.approx(0.96, abs=0.02)
        assert (in_tmp / "water_opt.xyz").exists() and (in_tmp / "water_opt.traj").exists()
        assert (in_tmp / "water_geometric.log").exists() and (in_tmp / "water_geometric_optim.xyz").exists()
        import ase.io

        frames = ase.io.read(in_tmp / "water_opt.traj", index=":")
        assert len(frames) > 1
        assert frames[-1].get_potential_energy() < frames[0].get_potential_energy()
        np.testing.assert_allclose(frames[-1].get_positions(), water.get_positions())

    def test_bond_constraint_is_kept(self, in_tmp, water):
        water.set_distance(0, 1, 1.2, fix=0)
        driver = self.make_driver(water)
        load_task("opt")(
            driver, {"optimizer": "geometric", "constraints": {"fix_bond": [[0, 1]]}, "max_steps": 60}, "water"
        )
        assert water.get_distance(0, 1) == pytest.approx(1.2, abs=1e-3)
        assert water.get_distance(0, 2) == pytest.approx(0.96, abs=0.03)

    def test_unconverged_run_still_writes_outputs(self, in_tmp, water):
        from qc_launcher.tasks.opt_geometric import run_opt_geometric

        water.set_distance(0, 1, 1.3, fix=0)
        driver = self.make_driver(water)
        assert run_opt_geometric(driver, {"max_steps": 1}, "water") is False
        assert (in_tmp / "water_opt.xyz").exists()
        import ase.io

        frames = ase.io.read(in_tmp / "water_opt.traj", index=":")
        assert len(frames) == 2  # initial geometry and the single step, recovered from geomeTRIC's _optim.xyz
        assert frames[-1].get_potential_energy() < frames[0].get_potential_energy()
        np.testing.assert_allclose(frames[-1].get_positions(), water.get_positions())

    def test_ts_search_uses_driver_hessian(self, in_tmp):
        import ase.io

        from qc_launcher.tasks.opt_geometric import run_opt_geometric

        da = ase.io.read(str(EXAMPLES / "3-transition_state" / "da.xyz"))
        driver = self.make_driver(da)
        assert run_opt_geometric(driver, {"ts": True, "max_steps": 60}, "da") is True
        assert (in_tmp / "da_geometric_hessian.txt").exists()
        hessian = driver.compute_hessian(da, use_cache=False, hess_format="ase")
        eigenvalues = np.linalg.eigvalsh(np.asarray(hessian).reshape(3 * len(da), 3 * len(da)))
        assert np.sum(eigenvalues < -0.1) == 1  # one imaginary mode (eV/Angstrom^2); trans/rot stay near zero
