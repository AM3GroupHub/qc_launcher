"""Benchmark configuration and failure reporting without quantum-chemistry calculations."""

import importlib.util
import json
from pathlib import Path

import pytest

from .conftest import write_xyz


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_refine.py"
    spec = importlib.util.spec_from_file_location("bench_refine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uppercase_mp2_uses_all_electrons_on_both_backends(in_tmp, water, monkeypatch, bench):
    write_xyz(in_tmp / "water.xyz", water)
    configs = []

    def run(driver, config, filename):
        configs.append(config)
        return {"e_hf": -1, "e_corr": -0.1, "e_total": -1.1, "timings": {}}

    monkeypatch.setattr(bench, "run_refine", run)
    assert bench.main(["water.xyz", "--methods", "MP2", "--backends", "PYSCF", "BYTEQC"]) == 0
    assert [c["backend"] for c in configs] == ["pyscf", "byteqc"]
    assert all(c["method"] == "mp2" and c["frozen_core"] is False for c in configs)


def test_failed_case_returns_failure_and_keeps_successful_rows(in_tmp, water, monkeypatch, bench):
    write_xyz(in_tmp / "water.xyz", water)

    def run(driver, config, filename):
        if config["backend"] == "byteqc":
            raise RuntimeError("backend failed")
        return {"e_hf": -1, "e_corr": -0.1, "e_total": -1.1, "timings": {}}

    monkeypatch.setattr(bench, "run_refine", run)
    assert bench.main(["water.xyz", "--methods", "mp2"]) == 1
    rows = json.loads((in_tmp / "water_bench_refine.json").read_text())
    assert len(rows) == 2
    assert rows[0]["e_total"] == -1.1
    assert "backend failed" in rows[1]["note"]
