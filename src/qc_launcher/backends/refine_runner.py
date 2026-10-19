#!/usr/bin/env python
"""Energy-refinement runner: RHF followed by MP2, CCSD or CCSD(T) with pyscf (CPU) or ByteQC (GPU).

qc_launcher executes this file in a separate interpreter (the one of the backend environment):

    python refine_runner.py input.json output.json

It must stay importable under old pyscf releases (ByteQC pins pyscf 2.5), so it uses only the standard
library, numpy, pyscf and, for ``backend == "byteqc"``, ByteQC. It never imports qc_launcher.

Input JSON keys: symbols, positions (Angstrom), charge, multiplicity, basis, auxbasis, ecp, backend
("pyscf" | "byteqc"), method ("mp2" | "ccsd" | "ccsd(t)"), frozen_core (bool | int), conv_tol,
max_cycle, scf ({conv_tol, max_cycle, density_fit, gpu, auxbasis}), ngpu, gpu_memory_gb, cpu_memory_gb, scratch_dir,
verbose, max_memory_mb.
Output JSON keys: e_hf, e_corr_mp2 / e_corr_ccsd / e_t (as applicable), e_corr, e_total (Eh), converged
flags, nao, nocc, n_frozen, timings (s), versions, warnings, and on failure error + traceback.
"""

import json
import os
import platform
import sys
import time
import traceback

# chemical-core orbital counts per element (H..Rn), used when pyscf lacks data.elements.chemcore
_CORE_BY_PERIOD = [(2, 0), (10, 1), (18, 5), (36, 9), (54, 18), (86, 27), (118, 43)]


def _n_core_orbitals(mol) -> int:
    try:
        from pyscf.data.elements import chemcore

        return int(chemcore(mol))
    except Exception:  # noqa: BLE001 - older pyscf
        n = 0
        for i in range(mol.natm):
            z = mol.atom_charge(i) + mol.atom_nelec_core(i)
            core = next(c for zmax, c in _CORE_BY_PERIOD if z <= zmax)
            n += max(core - mol.atom_nelec_core(i) // 2, 0)
        return n


def _versions(backend: str) -> dict:
    import numpy
    import pyscf

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.machine(),
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import cupy

        versions["cupy"] = cupy.__version__
        n = cupy.cuda.runtime.getDeviceCount()
        versions["gpus"] = [cupy.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in range(n)]
    except Exception:  # noqa: BLE001
        pass
    if backend == "byteqc":
        try:
            import byteqc

            versions["byteqc_path"] = os.path.dirname(byteqc.__file__)
            head = os.path.join(versions["byteqc_path"], ".git", "HEAD")
            if os.path.exists(head):
                with open(head) as handle:
                    ref = handle.read().strip()
                if ref.startswith("ref:"):
                    ref_path = os.path.join(versions["byteqc_path"], ".git", ref.split()[1])
                    if os.path.exists(ref_path):
                        with open(ref_path) as handle:
                            ref = handle.read().strip()
                versions["byteqc_commit"] = ref[:12]
        except Exception:  # noqa: BLE001
            pass
    return versions


def run(spec: dict) -> dict:
    scratch = spec.get("scratch_dir")
    if scratch:
        os.makedirs(scratch, exist_ok=True)
        os.environ["PYSCF_TMPDIR"] = scratch
    from pyscf import gto, scf

    backend = spec.get("backend", "pyscf")
    method = spec["method"].lower()
    out = {"backend": backend, "method": method, "warnings": [], "timings": {}}
    verbose = int(spec.get("verbose", 3))

    atom = [(symbol, tuple(pos)) for symbol, pos in zip(spec["symbols"], spec["positions"])]
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        basis=spec["basis"],
        ecp=spec.get("ecp"),
        charge=int(spec.get("charge", 0)),
        spin=int(spec.get("multiplicity", 1)) - 1,
        verbose=verbose,
        max_memory=spec.get("max_memory_mb") or 4000,
    )
    if mol.spin != 0:
        raise ValueError("The refinement backends are closed-shell (RHF-based) only.")
    out["nao"] = int(mol.nao)
    out["nocc"] = int(mol.nelectron // 2)

    scf_opts = spec.get("scf") or {}
    scf_density_fit = bool(scf_opts.get("density_fit", True))
    gpu_scf = bool(scf_opts.get("gpu", False))
    if gpu_scf and backend != "byteqc":
        out["warnings"].append("scf.gpu applies to the byteqc backend only; the SCF ran on the CPU.")
        gpu_scf = False
    if backend == "byteqc":
        from byteqc.lib import Mg

        Mg.set_gpus(int(spec.get("ngpu", 1)))
    t0 = time.time()
    if gpu_scf:
        # ByteQC's cuobc RHF runs the SCF with exact integrals on the GPU (its own usage pattern); a DF wrapper
        # is added afterwards, without re-running the SCF, so that the correlated step uses DF integrals.
        from byteqc.cuobc import scf as cuscf

        mf = cuscf.RHF(mol)
    else:
        mf = scf.RHF(mol)
        if scf_density_fit:
            mf = mf.density_fit(auxbasis=scf_opts.get("auxbasis"))
    mf.conv_tol = float(scf_opts.get("conv_tol", 1e-10))
    mf.max_cycle = int(scf_opts.get("max_cycle", 100))
    e_hf = float(mf.kernel())
    out["timings"]["scf_s"] = time.time() - t0
    out["e_hf"] = e_hf
    out["scf_converged"] = bool(mf.converged)
    out["scf_gpu"] = gpu_scf
    if not mf.converged:
        raise RuntimeError("The reference RHF did not converge.")
    if gpu_scf and scf_density_fit:
        mf = mf.density_fit(auxbasis=scf_opts.get("auxbasis"))

    frozen_setting = spec.get("frozen_core", True)
    if isinstance(frozen_setting, bool):
        n_frozen = _n_core_orbitals(mol) if frozen_setting else 0
    else:
        n_frozen = int(frozen_setting)
    out["n_frozen"] = n_frozen
    frozen = n_frozen or None
    auxbasis = spec.get("auxbasis")
    conv_tol = float(spec.get("conv_tol", 1e-8))
    max_cycle = int(spec.get("max_cycle", 100))

    if backend == "byteqc":
        from byteqc import cucc, cump2

        gpulim = spec.get("gpu_memory_gb")
        cpulim = spec.get("cpu_memory_gb")
        pool_kwargs = {
            "gpulim": None if gpulim is None else int(float(gpulim) * 1e9),
            "cpulim": None if cpulim is None else int(float(cpulim) * 1e9),
        }
        if scratch:
            pool_kwargs["path"] = scratch

    e_corr = 0.0
    if method == "mp2":
        t0 = time.time()
        if backend == "byteqc":
            if n_frozen:
                out["warnings"].append("ByteQC DF-MP2 has no frozen core; all electrons were correlated.")
                out["n_frozen"] = 0
            e_corr_mp2, _ = cump2.DFMP2(mol, mf, auxbasis=auxbasis, verbose=verbose)
        else:
            from pyscf.mp.dfmp2_native import DFMP2

            mp = DFMP2(mf, frozen=frozen, auxbasis=auxbasis)
            mp.verbose = verbose
            e_corr_mp2 = mp.kernel()
        out["e_corr_mp2"] = e_corr = float(e_corr_mp2)
        out["timings"]["mp2_s"] = time.time() - t0
    elif method in ("ccsd", "ccsd(t)"):
        t0 = time.time()
        if backend == "byteqc":
            mcc = cucc.CCSD(mf, frozen=frozen, **pool_kwargs)
        else:
            from pyscf import cc

            mcc = cc.CCSD(mf, frozen=frozen)
        mcc.verbose = verbose
        mcc.conv_tol = conv_tol
        mcc.max_cycle = max_cycle
        e_corr_ccsd, t1, t2 = mcc.kernel()
        out["e_corr_ccsd"] = float(e_corr_ccsd)
        out["ccsd_converged"] = bool(mcc.converged)
        out["timings"]["ccsd_s"] = time.time() - t0
        if not mcc.converged:
            raise RuntimeError("CCSD did not converge.")
        e_corr = float(e_corr_ccsd)
        if method == "ccsd(t)":
            t0 = time.time()
            if backend == "byteqc":
                from byteqc.cucc.ccsd_t import kernel as ccsd_t_kernel

                e_t = ccsd_t_kernel(mcc, mcc.ao2mo(), t1, t2)
            else:
                e_t = mcc.ccsd_t()
            out["e_t"] = float(e_t)
            out["timings"]["ccsd_t_s"] = time.time() - t0
            e_corr += float(e_t)
    else:
        raise ValueError(f"Unknown method '{method}'. Use mp2, ccsd or ccsd(t).")

    out["e_corr"] = e_corr
    out["e_total"] = e_hf + e_corr
    out["versions"] = _versions(backend)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: refine_runner.py input.json output.json", file=sys.stderr)
        return 2
    with open(argv[0]) as handle:
        spec = json.load(handle)
    try:
        result = run(spec)
        status = 0
    except Exception as exc:  # noqa: BLE001 - reported through the output file
        result = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        status = 1
    with open(argv[1], "w") as handle:
        json.dump(result, handle, indent=2)
    return status


if __name__ == "__main__":
    sys.exit(main())
