#!/usr/bin/env python
"""Compare refinement backends (pyscf CPU vs ByteQC GPU) on one XYZ file: energies, differences and timings.

    python scripts/bench_refine.py water.xyz --basis cc-pVDZ --methods mp2 ccsd "ccsd(t)" \
        --backends pyscf byteqc --python /path/to/byteqc-env/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import ase.io

from qc_launcher.tasks.refine import run_refine


class _Driver:
    def __init__(self, atoms):
        self.atoms = atoms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xyz")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--methods", nargs="+", default=["mp2", "ccsd", "ccsd(t)"])
    parser.add_argument("--backends", nargs="+", default=["pyscf", "byteqc"])
    parser.add_argument("--python", default=sys.executable, help="interpreter running the backends")
    parser.add_argument("--all-electron", action="store_true", help="no frozen core")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument(
        "--scf-gpu", action="store_true", help="run the reference SCF with ByteQC cuobc (byteqc backend)"
    )
    args = parser.parse_args(argv)

    atoms = ase.io.read(args.xyz)
    atoms.info.update(charge=args.charge, multiplicity=1)
    stem = Path(args.xyz).stem
    rows = []
    for method in args.methods:
        results = {}
        for backend in args.backends:
            # ByteQC DF-MP2 has no frozen core: when it takes part, compare all-electron MP2 on every backend
            frozen = not (args.all_electron or (method == "mp2" and "byteqc" in args.backends))
            config = {
                "backend": backend,
                "python": args.python,
                "method": method,
                "basis": args.basis,
                "frozen_core": frozen,
                "ngpu": args.ngpu,
                "scf": {"gpu": args.scf_gpu and backend == "byteqc"},
                "verbose": 3,
            }
            t0 = time.time()
            try:
                out = run_refine(_Driver(atoms), config, f"{stem}_{method.replace('(', '').replace(')', '')}_{backend}")
                results[backend] = out
                rows.append(
                    (
                        method,
                        backend,
                        out["e_hf"],
                        out["e_corr"],
                        out["e_total"],
                        out["timings"],
                        time.time() - t0,
                        "gpu scf" if out.get("scf_gpu") else "",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep benchmarking the other cases
                rows.append(
                    (
                        method,
                        backend,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        {},
                        time.time() - t0,
                        f"{type(exc).__name__}: {exc}"[:200],
                    )
                )
        if len(results) == 2:
            a, b = (results[k] for k in args.backends[:2])
            print(
                f"\n{method.upper()}: {args.backends[0]} - {args.backends[1]}: dE_HF = {a['e_hf'] - b['e_hf']:.3e}, dE_corr = {a['e_corr'] - b['e_corr']:.3e} Eh"
            )

    print(f"\n{stem} / {args.basis} ({len(atoms)} atoms)")
    print(
        f"{'method':<8}{'backend':<8}{'E_HF':>18}{'E_corr':>16}{'E_total':>18}{'t_scf':>8}{'t_corr':>8}{'t_wall':>8}  note"
    )
    for method, backend, e_hf, e_corr, e_tot, timings, wall, note in rows:
        t_corr = sum(v for k, v in timings.items() if k != "scf_s")
        print(
            f"{method:<8}{backend:<8}{e_hf:18.9f}{e_corr:16.9f}{e_tot:18.9f}{timings.get('scf_s', float('nan')):8.1f}{t_corr:8.1f}{wall:8.1f}  {note}"
        )
    Path(f"{stem}_bench_refine.json").write_text(
        json.dumps(
            [
                {
                    "method": r[0],
                    "backend": r[1],
                    "e_hf": r[2],
                    "e_corr": r[3],
                    "e_total": r[4],
                    "timings": r[5],
                    "wall_s": r[6],
                    "note": r[7],
                }
                for r in rows
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
