"""Energy refinement: a high-level single-point energy (MP2, CCSD, CCSD(T)) on the current geometry.

The correlated calculation runs in a separate interpreter through ``backends/refine_runner.py`` so that
backends with incompatible dependency pins (ByteQC needs pyscf 2.5) can live in their own environment.
When thermochemistry from ``freq`` is available, composite values E_0K / H / G are formed as
``X_refined = X - E0 + E_refined``, i.e. the cheap driver's thermal corrections on top of the refined energy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from ase.units import Hartree

from qc_launcher.drivers import BaseDriver

from .common import timed

BACKENDS = ("pyscf", "byteqc")
METHODS = ("mp2", "ccsd", "ccsd(t)")
COMPOSITE_KEYS = ("E_0K", "H_tot", "G_tot", "H_tot_qrrho", "G_tot_qrrho")


def runner_path() -> str:
    return str(resources.files("qc_launcher.backends").joinpath("refine_runner.py"))


def _thermo_value(thermo: dict, key: str) -> float | None:
    """Read a thermo quantity from run_freq's dict ((value, unit) tuples) or from the thermo JSON layout."""
    entry = thermo.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return float(entry["value"])
    return float(entry[0])


def composite_thermochemistry(thermo: dict, e_refined: float) -> dict[str, float]:
    """Replace the driver's electronic energy E0 by the refined energy in the thermochemical totals (Eh)."""
    e0 = _thermo_value(thermo, "E0")
    if e0 is None:
        return {}
    composite = {}
    for key in COMPOSITE_KEYS:
        value = _thermo_value(thermo, key)
        if value is not None:
            composite[f"{key}_refined"] = value - e0 + e_refined
    return composite


def probe_backend(python: str, backend: str, *, env: dict[str, str] | None = None) -> None:
    """Fail early when the interpreter cannot import what the backend needs."""
    module = "byteqc.cucc, byteqc.cump2" if backend == "byteqc" else "pyscf"
    result = subprocess.run(
        [python, "-c", f"import {module}"], capture_output=True, text=True, env=env, timeout=600, check=False
    )
    if result.returncode != 0:
        hint = (
            " (set refine.python to the ByteQC environment's interpreter and refine.pythonpath to a directory holding the ByteQC clone)"
            if backend == "byteqc"
            else ""
        )
        raise RuntimeError(f"{python} cannot import {module}{hint}:\n{result.stderr.strip()[-2000:]}")


def run_refine(driver: BaseDriver, config: dict, filename: str = "molecule", thermo: dict | None = None) -> dict:
    """Config keys (defaults): ``backend`` (pyscf | byteqc), ``python`` (interpreter of the backend env;
    default: the current one), ``pythonpath`` (prepended for the runner), ``method`` (ccsd(t) | ccsd | mp2),
    ``basis`` (cc-pVTZ), ``auxbasis`` (pyscf default), ``ecp``, ``frozen_core`` (True; or an orbital count),
    ``conv_tol`` (1e-8), ``max_cycle`` (100), ``scf`` ({conv_tol: 1e-10, max_cycle: 100, density_fit: True,
    gpu: False, auxbasis}), ``ngpu`` (1), ``gpu_memory_gb``, ``cpu_memory_gb``, ``scratch_dir``, ``max_memory_mb``,
    ``verbose`` (3), ``timeout_s``, ``outputfile`` ({filename}_refine.json).

    Returns the runner's results plus the composite thermochemistry; closed-shell systems only.
    """
    with timed("Energy refinement"):
        atoms = driver.atoms
        backend = str(config.get("backend", "pyscf")).lower()
        method = str(config.get("method", "ccsd(t)")).lower()
        if backend not in BACKENDS:
            raise ValueError(f"refine.backend '{backend}' is not valid. Use one of {BACKENDS}.")
        if method not in METHODS:
            raise ValueError(f"refine.method '{method}' is not valid. Use one of {METHODS}.")
        if int(atoms.info.get("multiplicity", 1)) != 1:
            raise ValueError("Energy refinement supports closed-shell (multiplicity 1) systems only.")
        python = str(config.get("python") or sys.executable)
        if not os.path.exists(python):
            raise ValueError(f"refine.python '{python}' does not exist.")
        env = dict(os.environ)
        if config.get("pythonpath"):
            env["PYTHONPATH"] = os.pathsep.join([str(config["pythonpath"]), env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep
            )
        probe_backend(python, backend, env=env)

        spec: dict[str, Any] = {
            "backend": backend,
            "method": method,
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(),
            "charge": int(atoms.info.get("charge", 0)),
            "multiplicity": int(atoms.info.get("multiplicity", 1)),
            "basis": config.get("basis", "cc-pVTZ"),
            "auxbasis": config.get("auxbasis"),
            "ecp": config.get("ecp"),
            "frozen_core": config.get("frozen_core", True),
            "conv_tol": float(config.get("conv_tol", 1e-8)),
            "max_cycle": int(config.get("max_cycle", 100)),
            "scf": dict(config.get("scf") or {}),
            "ngpu": int(config.get("ngpu", 1)),
            "gpu_memory_gb": config.get("gpu_memory_gb"),
            "cpu_memory_gb": config.get("cpu_memory_gb"),
            "scratch_dir": config.get("scratch_dir"),
            "max_memory_mb": config.get("max_memory_mb"),
            "verbose": int(config.get("verbose", 3)),
        }
        input_path = Path(f"{filename}_refine_input.json")
        raw_path = Path(f"{filename}_refine_raw.json")
        log_path = Path(f"{filename}_refine.log")
        input_path.write_text(json.dumps(spec, indent=2))
        raw_path.unlink(missing_ok=True)

        timeout = config.get("timeout_s")
        print(f"Refinement: {method.upper()}/{spec['basis']} with backend '{backend}' ({python}); log: {log_path}")
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [python, runner_path(), str(input_path), str(raw_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=None if timeout is None else float(timeout),
                check=False,
            )
        if not raw_path.exists():
            tail = log_path.read_text()[-2000:] if log_path.exists() else ""
            raise RuntimeError(f"The refinement runner produced no output (exit code {proc.returncode}).\n{tail}")
        results = json.loads(raw_path.read_text())
        if "error" in results:
            raise RuntimeError(
                f"The refinement runner failed: {results['error']}\n{results.get('traceback', '')[-2000:]}"
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"The refinement runner failed (exit code {proc.returncode}).\n{log_path.read_text()[-2000:]}"
            )

        for warning in results.get("warnings", []):
            print(f"Warning: {warning}")
        print(f"Refined E_HF    [Eh]: {results['e_hf']:16.10f}")
        for key, label in (
            ("e_corr_mp2", "MP2 correlation"),
            ("e_corr_ccsd", "CCSD correlation"),
            ("e_t", "(T) correction"),
        ):
            if key in results:
                print(f"Refined {label:<16}[Eh]: {results[key]:16.10f}")
        print(f"Refined E_total [Eh]: {results['e_total']:16.10f}   ({results['e_total'] * Hartree:.6f} eV)")

        if thermo is None:
            thermo_path = Path(f"{filename}_thermo.json")
            if thermo_path.exists():
                thermo = json.loads(thermo_path.read_text())
        composite = composite_thermochemistry(thermo, results["e_total"]) if thermo else {}
        for key, value in composite.items():
            print(f"{key:<20}[Eh]: {value:16.10f}")
        if thermo and not composite:
            print("Note: no E0 in the thermochemistry data; composite energies not formed.")

        output = {**results, "config": {k: v for k, v in config.items()}, "composite": composite}
        outputfile = config.get("outputfile", f"{filename}_refine.json")
        Path(outputfile).write_text(json.dumps(output, indent=2))
        return output
