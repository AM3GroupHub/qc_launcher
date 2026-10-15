from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

# marker name -> importable package that must be present, otherwise the test is skipped
MARKER_PACKAGES = {
    "pyscf": "pyscf",
    "gpu": "cupy",
    "tblite": "tblite",
    "mace": "mace",
    "uma": "fairchem",
    "sella": "sella",
    "pygsm": "pygsm",
    "pysisyphus": "pysisyphus",
}

DEFAULT_MODEL_PATHS = {
    "mace": "/home/admin/storage/models/mace/mace-mh-1.model",
    "uma": "/home/admin/storage/models/uma/uma-s-1p2.pt",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, package in MARKER_PACKAGES.items():
            if item.get_closest_marker(marker) and importlib.util.find_spec(package) is None:
                item.add_marker(pytest.mark.skip(reason=f"{package} is not installed"))


def _model_path(kind: str) -> str:
    path = os.environ.get(f"QC_LAUNCHER_{kind.upper()}_MODEL", DEFAULT_MODEL_PATHS[kind])
    if not Path(path).exists():
        pytest.skip(f"{kind} model file not found: {path}")
    return path


@pytest.fixture(scope="session")
def mace_model_path() -> str:
    return _model_path("mace")


@pytest.fixture(scope="session")
def uma_model_path() -> str:
    return _model_path("uma")


@pytest.fixture
def water() -> Atoms:
    atoms = molecule("H2O")
    atoms.info["charge"] = 0
    atoms.info["multiplicity"] = 1
    return atoms


@pytest.fixture
def in_tmp(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_xyz(path: Path, atoms: Atoms) -> Path:
    import ase.io

    ase.io.write(path, atoms, columns=["symbols", "positions"])
    return path


def write_yaml(path: Path, data: dict) -> Path:
    import yaml

    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
