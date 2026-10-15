import ase.io
import pytest
from ase import Atoms

from qc_launcher.tasks import load_task

pytestmark = [pytest.mark.tblite, pytest.mark.slow]


@pytest.fixture
def hcn_hnc():
    hcn = Atoms("HCN", positions=[(-1.07, 0, 0), (0, 0, 0), (1.16, 0, 0)])
    hnc = Atoms("HCN", positions=[(2.16, 0, 0), (0, 0, 0), (1.16, 0, 0)])  # H on the N side
    for frame in (hcn, hnc):
        frame.info.update(charge=0, multiplicity=1)
    return [hcn, hnc]


def test_neb_runs_and_writes_results(in_tmp, hcn_hnc):
    from qc_launcher.drivers.tblite_driver import TBLiteDriver

    driver = TBLiteDriver(hcn_hnc[0], {"verbosity": 0}, verbose=False)
    config = {"num_images": 3, "fmax": 0.5, "max_neb_steps": 10, "max_opt_steps": 3, "climb": True, "climb_after": 5}
    load_task("neb")(driver, config, hcn_hnc, "hcn")
    images = ase.io.read(in_tmp / "hcn_neb.xyz", index=":")
    assert len(images) == 5
    assert (in_tmp / "hcn_ts.xyz").exists()
