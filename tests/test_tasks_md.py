import ase.io
import pytest

from qc_launcher import cli
from qc_launcher.utils.config import ConfigDeprecationWarning

from .conftest import write_xyz, write_yaml

pytestmark = pytest.mark.tblite


def test_md_frames_and_resume(in_tmp, water):
    write_xyz(in_tmp / "water.xyz", water)
    md = {
        "ensemble": "NVT",
        "minimize": False,
        "timestep": 0.5,
        "nsteps": 20,
        "loginterval": 10,
        "dtamp": 25.0,
        "logfile": "md.log",
    }
    config = {"inputfile": "water.xyz", "driver": {"name": "tblite", "verbosity": 0}, "md": md}
    write_yaml(in_tmp / "run.yaml", config)
    with pytest.warns(ConfigDeprecationWarning, match="dtamp"):
        cli.main(["md", "run.yaml"])
    assert len(ase.io.Trajectory("water_md.traj")) == 3  # steps 0, 10, 20

    md.update(tdamp=md.pop("dtamp"), nsteps=40, resume=True)
    write_yaml(in_tmp / "run.yaml", config)
    cli.main(["md", "run.yaml"])
    assert len(ase.io.Trajectory("water_md.traj")) == 5  # exactly 20 more steps appended

    cli.main(["md", "run.yaml"])  # nothing left to do
    assert len(ase.io.Trajectory("water_md.traj")) == 5


def test_fixed_atom_zero_is_honoured(in_tmp, water):
    water.set_tags([1, 0, 0])  # only atom 0 fixed: np.any(indices) used to drop this case
    write_xyz(in_tmp / "water.xyz", water)
    import ase.io as aio

    frame = aio.read(in_tmp / "water.xyz")
    frame.set_tags([1, 0, 0])
    aio.write(in_tmp / "water.xyz", frame)
    md = {"ensemble": "NVE", "minimize": False, "timestep": 0.5, "nsteps": 10, "loginterval": 10, "logfile": "md.log"}
    write_yaml(in_tmp / "run.yaml", {"inputfile": "water.xyz", "driver": {"name": "tblite", "verbosity": 0}, "md": md})
    cli.main(["md", "run.yaml"])
    traj = ase.io.Trajectory("water_md.traj")
    assert (traj[0].positions[0] == traj[-1].positions[0]).all()
