#!/usr/bin/env python
"""Time native gpu4pyscf against the cuEST accelerator for one structure and driver config.

Usage:
    python scripts/bench_accelerator.py run.yaml [--modes none cuest:native_fp64 cuest:default] [--no-grad]

`run.yaml` is a qc_launch config (``inputfile`` + ``driver`` block). For each mode the driver is built twice:
the first build/SCF includes cuEST's JIT warm-up, the second one is reported as the "warm" SCF time. Energies
and forces of every mode are compared with the native run. Keep the numbers to yourself: cuEST's license does
not allow publishing benchmark results without NVIDIA's permission.
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
from ase.units import Hartree

from qc_launcher.cli import load_structures, load_yaml_config
from qc_launcher.drivers import driver_from_config


def timed(func):
    start = time.time()
    value = func()
    return value, time.time() - start


def run_mode(config, atoms, mode, with_grad):
    block = copy.deepcopy(config["driver"])
    if mode == "none":
        block["accelerator"] = "none"
    else:
        _, math_mode = mode.split(":")
        block["accelerator"] = "cuest"
        block.setdefault("cuest", {})["math_mode"] = math_mode
    driver, t_build = timed(lambda: driver_from_config(block, atoms.copy(), verbose=False))
    energy, t_scf_first = timed(driver.compute_energy)
    forces, t_grad = timed(driver.compute_forces) if with_grad else (None, float("nan"))
    warm_driver = driver_from_config(block, atoms.copy(), verbose=False)
    _, t_scf_warm = timed(warm_driver.compute_energy)
    return {
        "mode": mode,
        "build_s": t_build,
        "scf_first_s": t_scf_first,
        "scf_warm_s": t_scf_warm,
        "grad_s": t_grad,
        "energy_eh": energy / Hartree,
        "forces": forces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--modes", nargs="+", default=["none", "cuest:native_fp64", "cuest:default"])
    parser.add_argument("--no-grad", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    atoms, _ = load_structures(config)
    rows = [run_mode(config, atoms, mode, not args.no_grad) for mode in args.modes]
    reference = rows[0]
    print(f"\n{len(atoms)} atoms, {config['driver'].get('xc', 'B3LYP')}/{config['driver'].get('basis', 'def2-SVP')}")
    print(
        f"{'mode':<20}{'build [s]':>11}{'SCF 1st [s]':>13}{'SCF warm [s]':>14}{'grad [s]':>10}{'dE vs none [Eh]':>18}{'max|dF| [eV/A]':>16}"
    )
    for row in rows:
        de = row["energy_eh"] - reference["energy_eh"]
        df = float("nan") if row["forces"] is None else float(np.abs(row["forces"] - reference["forces"]).max())
        print(
            f"{row['mode']:<20}{row['build_s']:>11.2f}{row['scf_first_s']:>13.2f}{row['scf_warm_s']:>14.2f}"
            f"{row['grad_s']:>10.2f}{de:>18.2e}{df:>16.1e}"
        )


if __name__ == "__main__":
    main()
