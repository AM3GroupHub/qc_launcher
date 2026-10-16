import json

import numpy as np
import pytest

from qc_launcher.tasks.freq import resolve_bav, run_freq

from .helpers import FakeDriver

pytestmark = pytest.mark.pyscf


def test_run_freq_returns_thermo_and_writes_json(in_tmp, water):
    driver = FakeDriver(water, {"k": 5.0})
    results = run_freq(driver, {"Bav": "1e-44", "temperature": 300.0}, "water")  # YAML gives '1e-44' as a string
    assert "G_tot" in results and "G_tot_qrrho" in results
    assert results["G_tot_qrrho"][0] == pytest.approx(results["H_tot_qrrho"][0] - 300.0 * results["S_tot_qrrho"][0])
    payload = json.loads((in_tmp / "water_thermo.json").read_text())
    assert len(payload["frequencies_cm1"]) == 3  # 3N - 6
    assert payload["qrrho_parameters"]["Bav_kg_m2"] == 1e-44
    assert payload["G_tot_qrrho"]["unit"] == "Eh"
    assert (in_tmp / "water_vib.txt").read_text().splitlines()[0] == "3 3"


def test_run_freq_without_qrrho_and_bav_auto(in_tmp, water):
    driver = FakeDriver(water, {"k": 5.0})
    plain = run_freq(driver, {"qrrho": False}, "water")
    assert "G_tot_qrrho" not in plain
    auto = run_freq(driver, {"Bav": "auto"}, "water")
    bav = json.loads((in_tmp / "water_thermo.json").read_text())["qrrho_parameters"]["Bav_kg_m2"]
    assert 1e-48 < bav < 1e-45  # Tr(I)/3 of water is ~2e-47 kg m^2
    assert auto["S_vib_qrrho"][0] != pytest.approx(plain["S_vib"][0])


def test_resolve_bav_rejects_garbage():
    with pytest.raises(ValueError, match="Invalid Bav"):
        resolve_bav("later", None, np.array([1.0, 1.0, 1.0]))
    assert resolve_bav(0.0, None, np.array([1.0, 1.0, 1.0])) == 1e-50
