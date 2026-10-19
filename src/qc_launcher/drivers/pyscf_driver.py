"""PySCF / gpu4pyscf driver: Kohn-Sham DFT energies, gradients and Hessians, optionally on the GPU.

Optional acceleration of the GPU path with NVIDIA cuEST through gpu4pyscf's ``cuest_wrapper``
(``accelerator: cuest``). cuEST provides density-fitted J/K, XC/VV10, PCM and ECP matrices and their
nuclear gradients but no analytic Hessian, so Hessians are built from central differences of the accelerated
gradients by default (``numerical_hessian: false`` switches to a native gpu4pyscf Hessian at the same geometry).
"""

from __future__ import annotations

import atexit
import os
import warnings
import weakref
from types import MethodType

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from pyscf import dft, gto, lib

from qc_launcher.utils.config import ConfigDeprecationWarning, rename_deprecated_keys
from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency
from qc_launcher.utils.utils import read_pcm_eps

from .base_driver import BaseDriver

PCM_METHODS = ("c-pcm", "ief-pcm", "ss(v)pe", "cosmo")
ACCELERATORS = ("none", "cuest")
CUEST_SOLVATION_MODELS = ("c-pcm", "cpcm", "cosmo")  # the only PCM variants the gpu4pyscf cuEST wrapper accepts
CUEST_PARTS = ("hcore", "jk", "xc", "nlc", "pcm")  # parts of the SCF that can be sent back to native gpu4pyscf
CUEST_MATH_MODES = ("native_fp64", "default")


def _asnumpy(array) -> np.ndarray:
    """Return a NumPy array for either a NumPy or a CuPy input."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _gpu_name() -> str:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)["name"].decode()
    except Exception:  # noqa: BLE001 - purely informational
        return "unknown GPU"


_CUEST_DRIVERS: weakref.WeakSet = weakref.WeakSet()
_INERT_BUNDLE_CLASS = None


def _release_cuest_handles() -> None:
    """Destroy cuEST handles of still-alive drivers while the bindings are importable.

    gpu4pyscf's HandleBundle frees its plans and the cuEST handle in __del__, which at interpreter shutdown
    can run after the cuest.bindings module has been torn down and then prints an AttributeError traceback.
    After freeing everything here the bundle's destructor is turned into a no-op.
    """
    global _INERT_BUNDLE_CLASS
    for driver in list(_CUEST_DRIVERS):
        handles = getattr(getattr(driver, "method", None), "handles", None)
        if handles is None or type(handles) is _INERT_BUNDLE_CLASS:
            continue
        try:
            import cuest.bindings as ce
            from gpu4pyscf.lib.cuest_wrapper import HandleBundle

            if _INERT_BUNDLE_CLASS is None:

                class InertHandleBundle(HandleBundle):
                    def __del__(self):
                        pass

                _INERT_BUNDLE_CLASS = InertHandleBundle
            handles.reset()
            ce.cuestDestroy(handle=handles.cuest_handle)
            handles.__class__ = _INERT_BUNDLE_CLASS
        except Exception:  # noqa: BLE001 - best effort at shutdown
            pass


atexit.register(_release_cuest_handles)


def _match_cuest_grids(mf) -> None:
    """Give a native gpu4pyscf method the grid and PCM conventions cuEST uses, so the two can be compared.

    cuEST partitions the molecular grid with the Stratmann scheme without atomic-radii adjustment and
    discretises PCM surfaces with ISWIG.
    """
    from pyscf.dft.gen_grid import stratmann

    for grids in (getattr(mf, "grids", None), getattr(mf, "nlcgrids", None)):
        if grids is not None:
            grids.becke_scheme = stratmann
            grids.radii_adjust = None
    if getattr(mf, "with_solvent", None) is not None:
        mf.with_solvent.surface_discretization_method = "ISWIG"


def _native_hcore_gradient(grad, dm) -> np.ndarray:
    """Use gpu4pyscf's current core-gradient API when the cuEST fallback uses removed imports."""
    import cupy

    if dm.ndim == 3:
        dm = dm.sum(axis=0)
    # _hcore_energy includes the overlap response; a zero energy-weighted density leaves just hcore.
    return grad._hcore_energy(dm, cupy.zeros_like(dm))


def _parse_3c(xc: str) -> tuple[str, str, str, str | None, str, str]:
    """Split a composite '3c' functional (e.g. B97-3c) into (xc, nlc, basis, ecp, xc_disp, xc_gcp)."""
    from gpu4pyscf.drivers.dft_3c_driver import parse_3c

    pyscf_xc, nlc, basis, ecp, (xc_disp, _), xc_gcp = parse_3c(xc.lower().replace("-3c", "3c"))
    return pyscf_xc, nlc, basis, ecp, xc_disp, xc_gcp


def _central_difference_hessian(mol, gradient_at, displacement: float) -> np.ndarray:
    """Symmetrised (N, N, 3, 3) Hessian in Eh/Bohr^2 from central differences of ``gradient_at(coords_bohr)``."""
    coords0 = mol.atom_coords(unit="Bohr").copy()
    natm = mol.natm
    hessian = np.zeros((natm, natm, 3, 3))
    for i in range(natm):
        for j in range(3):
            coords = coords0.copy()
            coords[i, j] += displacement
            grad_plus = gradient_at(coords)
            coords[i, j] -= 2.0 * displacement
            grad_minus = gradient_at(coords)
            hessian[i, :, j, :] = (grad_plus - grad_minus) / (2.0 * displacement)
    return 0.5 * (hessian + hessian.transpose(1, 0, 3, 2))


def _finite_difference_hessian(grad_method, displacement: float) -> np.ndarray:
    """Central-difference Hessian from a pyscf or gpu4pyscf gradient method; ``displacement`` in Bohr.

    The molecule is displaced in place and restored afterwards; the caller should reset its mean-field
    object because the scanner re-points the shared grids / DF objects.
    """
    scanner = grad_method.as_scanner()
    mol = grad_method.mol
    coords0 = mol.atom_coords(unit="Bohr").copy()

    def gradient_at(coords):
        mol.set_geom_(coords, unit="Bohr")
        _, gradient = scanner(mol)
        if not scanner.converged:
            raise RuntimeError("SCF did not converge at a displaced geometry during the finite-difference Hessian.")
        return _asnumpy(gradient)

    try:
        return _central_difference_hessian(mol, gradient_at, displacement)
    finally:
        mol.set_geom_(coords0, unit="Bohr")


class PySCFCalculator(Calculator):
    """ASE calculator that delegates to a :class:`PySCFDriver`, so both share one SCF state."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, driver: PySCFDriver, max_unconverged_steps: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.driver = driver
        self.max_unconverged_steps = max_unconverged_steps
        self.num_unconverged = 0

    def calculate(self, atoms: Atoms | None = None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or self.implemented_properties, system_changes)
        energy = self.driver.run_kernel(self.atoms if atoms is None else atoms)  # Eh
        if not self.driver.method.converged:
            self.num_unconverged += 1
            if self.max_unconverged_steps is not None and self.num_unconverged > self.max_unconverged_steps:
                print(f"Warning: SCF has not converged for {self.num_unconverged} steps")
        gradient = _asnumpy(self.driver.build_gradient_method().kernel())
        self.results = {"energy": energy * Hartree, "forces": -gradient * (Hartree / Bohr)}


class PySCFDriver(BaseDriver):
    """Kohn-Sham DFT with pyscf (CPU) or gpu4pyscf (GPU), density fitting and PCM/SMD solvation.

    Config keys (defaults): ``xc`` (B3LYP; names ending in ``3c`` select a composite method via gpu4pyscf),
    ``basis`` (def2-SVP), ``ecp``, ``restricted`` (None: by spin), ``nlc`` (''), ``dispersion``, ``grids`` /
    ``nlcgrids`` ({atom_grid: ...} or {level: n}), ``verbosity`` (2), ``max_memory`` [GB], ``threads``,
    ``scf_conv_tol`` (1e-8), ``direct_scf_tol`` (1e-8), ``scf_max_iter`` (50), ``level_shift`` (0),
    ``diis_space`` (8), ``with_df`` (True), ``auxbasis``, ``with_gpu`` (True), ``solvation`` ({model: ief-pcm |
    c-pcm | ss(v)pe | cosmo | smd, solvent | epsilon | solvent_descriptors}), ``retry_soscf`` (False),
    ``max_unconverged_steps``, ``numerical_hessian`` (False; True by default with cuEST), ``finite_diff_step``
    (5e-3 Angstrom), ``accelerator`` (none | cuest) and the ``cuest`` block ({math_mode: native_fp64 | default,
    threshold_pq (1e-14), density_fitting_cutoff (1e-12), max_workspace_gb (10), disable: [hcore | jk | xc | nlc |
    pcm], match_native_grids (False)}).

    cuEST needs ``with_df`` and ``with_gpu``, accepts only C-PCM / COSMO solvation, no ``retry_soscf`` and no
    composite ``3c`` methods.
    """

    def __init__(self, atoms: Atoms | None, config: dict, **kwargs):
        super().__init__(atoms=atoms, config=config, **kwargs)
        xc = str(self.config.get("xc", "B3LYP"))
        self.is_3c = xc.lower().endswith("3c")
        self.retry_soscf = bool(self.config.get("retry_soscf", False))
        self.accelerator = str(self.config.get("accelerator", "none")).lower()
        if self.accelerator not in ACCELERATORS:
            raise ValueError(f"Unknown accelerator '{self.accelerator}'. Use one of {ACCELERATORS}.")
        if self.accelerator == "cuest":
            self._check_cuest_config(self.config)  # before anything imports gpu4pyscf or cuEST
        self._3c = _parse_3c(xc) if self.is_3c else None
        self._dm_cache = None
        self._native_method = None  # plain gpu4pyscf method used for analytic Hessians under cuEST
        self.method = None if atoms is None else self.build_method()

    # ------------------------------------------------------------------ construction
    def _build_mol(self, config: dict) -> gto.Mole:
        threads = config.get("threads", os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        lib.num_threads(int(threads))
        max_memory = config.get("max_memory")
        return gto.M(
            atom=[(symbol, pos) for symbol, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())],
            basis=config.get("basis", "def2-SVP"),
            ecp=config.get("ecp"),
            charge=int(self.atoms.info.get("charge", 0)),  # plain ints: ASE may hand us numpy integers
            spin=int(self.atoms.info.get("multiplicity", 1)) - 1,
            max_memory=None if max_memory is None else float(max_memory) * 1024,  # GB -> MB
            verbose=int(config.get("verbosity", 2)),
        )

    def _build_mf(self, config: dict):
        mol = self._build_mol(config)
        xc = config.get("xc", "B3LYP")
        restricted = config.get("restricted")
        if restricted is True:
            mf = dft.RKS(mol, xc=xc) if mol.spin == 0 else dft.ROKS(mol, xc=xc)
        elif restricted is False:
            mf = dft.UKS(mol, xc=xc)
        else:
            mf = dft.KS(mol, xc=xc)
        nlc = config.get("nlc", "")
        mf.nlc = nlc
        mf.disp = config.get("dispersion")
        grids = config.get("grids")
        if grids:
            if "atom_grid" in grids:
                mf.grids.atom_grid = grids["atom_grid"]
            elif "level" in grids:
                mf.grids.level = grids["level"]
        nlcgrids = config.get("nlcgrids")
        if nlcgrids and (nlc or mf._numint.libxc.is_nlc(mf.xc)):
            if "atom_grid" in nlcgrids:
                mf.nlcgrids.atom_grid = nlcgrids["atom_grid"]
            if "level" in nlcgrids:
                mf.nlcgrids.level = nlcgrids["level"]
        if config.get("with_df", True):
            mf = mf.density_fit(auxbasis=config.get("auxbasis"))
        if config.get("with_gpu", True):
            try:
                import cupy
            except ImportError:
                self.log("GPU support is not available. Proceeding with CPU.")
            else:
                cupy.get_default_memory_pool().free_all_blocks()
                mf = mf.to_gpu()
        solvation = config.get("solvation")
        with_solvent = config.get("with_solvent")
        if with_solvent is not None:
            warnings.warn(
                "[driver] key 'with_solvent' is deprecated; a 'solvation' block alone enables solvation.",
                ConfigDeprecationWarning,
            )
            if with_solvent and solvation is None:
                solvation = {"model": "ief-pcm", "epsilon": 78.3553}
            elif not with_solvent:
                solvation = None
        if solvation:
            mf = self._apply_solvent(mf, solvation)
        mf.conv_tol = float(config.get("scf_conv_tol", 1e-8))
        mf.direct_scf_tol = float(config.get("direct_scf_tol", 1e-8))
        mf.max_cycle = int(config.get("scf_max_iter", 50))
        mf.level_shift = float(config.get("level_shift", 0.0))
        mf.diis_space = int(config.get("diis_space", 8))
        mf.chkfile = None
        return self._apply_accelerator(mf, config)

    # ------------------------------------------------------------------ cuEST
    def _check_cuest_config(self, config: dict) -> None:
        """Reject settings the gpu4pyscf cuEST wrapper cannot handle, before anything touches the GPU."""
        if not config.get("with_df", True):
            raise ValueError("accelerator 'cuest' requires density fitting (with_df: true).")
        if not config.get("with_gpu", True):
            raise ValueError("accelerator 'cuest' requires with_gpu: true.")
        if self.is_3c:
            raise ValueError("accelerator 'cuest' does not support composite '3c' methods.")
        if config.get("retry_soscf", False):
            raise ValueError("accelerator 'cuest' does not support retry_soscf (cuEST has no second-order SCF).")
        solvation = config.get("solvation")
        if config.get("with_solvent") is False:
            solvation = None
        elif solvation is None and config.get("with_solvent"):
            solvation = {"model": "ief-pcm"}
        if solvation:
            model = str(solvation.get("model", solvation.get("method", "ief-pcm"))).lower()
            if model not in CUEST_SOLVATION_MODELS:
                raise ValueError(f"accelerator 'cuest' supports only C-PCM and COSMO solvation, not '{model}'.")
        options = config.get("cuest") or {}
        unknown = {str(part).lower() for part in options.get("disable") or []} - set(CUEST_PARTS)
        if unknown:
            raise ValueError(f"cuest.disable: unknown parts {sorted(unknown)}. Use any of {CUEST_PARTS}.")
        math_mode = str(options.get("math_mode", "native_fp64")).lower().replace("-", "_")
        if math_mode not in CUEST_MATH_MODES:
            raise ValueError(f"cuest.math_mode '{math_mode}' is not valid. Use one of {CUEST_MATH_MODES}.")

    def _apply_accelerator(self, mf, config: dict):
        if str(config.get("accelerator", "none")).lower() != "cuest":
            return mf
        if "gpu4pyscf" not in type(mf).__module__:
            raise RuntimeError(
                "accelerator 'cuest' needs the GPU build of the method; cupy / gpu4pyscf are not available."
            )
        try:
            from gpu4pyscf.lib.cuest_wrapper import apply_cuest_wrapper
        except ModuleNotFoundError as exc:
            if is_missing_package(exc, "cuest"):
                raise missing_optional_dependency(
                    "accelerator 'cuest'", "cuest-cu13", package_name="nvidia-cuest-cu13 (nvidia-cuest-cu12 on CUDA 12)"
                ) from exc
            raise

        options = dict(config.get("cuest") or {})
        if options.get("match_native_grids", False):
            _match_cuest_grids(mf)

        if mf.with_df.auxbasis is None:
            from pyscf.df.addons import make_auxbasis

            # Native DF resolves this lazily, but cuEST requires an explicit basis before its first SCF.
            mf.with_df.auxbasis = make_auxbasis(mf.mol)
        mf = apply_cuest_wrapper(mf)
        # The wrapper defines this mutable cache on HandleBundle's class. Keep it per handle so one
        # driver's exchange settings cannot mark another driver's integral plan as reusable.
        mf.handles.dfintplan_parameter_cache = dict(mf.handles.dfintplan_parameter_cache)
        # The wrapper rebuilds its helpers on changes to the disable flags and thresholds.
        # Set the workspace first so those rebuilds propagate it to the helpers as well.
        mf.maximum_workspace_bytes = int(float(options.get("max_workspace_gb", 10)) * 1e9)
        for part in options.get("disable") or []:
            setattr(mf, f"turn_on_cuest_{str(part).lower()}", False)
        mf.density_fitting_cutoff = float(options.get("density_fitting_cutoff", 1e-12))
        mf.threshold_pq = float(options.get("threshold_pq", 1e-14))
        mf.math_mode = str(options.get("math_mode", "native_fp64")).lower().replace("-", "_")
        _CUEST_DRIVERS.add(self)
        self.log(
            f"cuEST accelerator on {_gpu_name()}: math_mode={mf.math_mode}, threshold_pq={mf.threshold_pq:g}, "
            f"workspace={mf.maximum_workspace_bytes / 1e9:g} GB, disabled={list(options.get('disable') or [])}"
        )
        return mf

    def _native_method_for_hessian(self):
        """A plain gpu4pyscf method at the current geometry, converged from the accelerated density."""
        if self._native_method is None:
            self._native_method = self._build_mf({**self.config, "accelerator": "none"})
            if (self.config.get("cuest") or {}).get("match_native_grids", False):
                _match_cuest_grids(self._native_method)
        native = self._native_method
        native.reset(self.method.mol.copy())
        native.kernel(dm0=self.method.make_rdm1())
        if not native.converged:
            raise RuntimeError("The native gpu4pyscf SCF for the analytic Hessian did not converge.")
        return native

    def _finite_difference_hessian_cuest(self, displacement: float) -> np.ndarray:
        """Central differences of the cuEST gradients, driving the wrapped method directly (no scanner)."""
        mol0 = self.method.mol
        coords0 = mol0.atom_coords(unit="Bohr").copy()
        dm0 = self.method.make_rdm1()

        def gradient_at(coords):
            self._reset_method(mol0.set_geom_(coords, unit="Bohr", inplace=False))
            self.method.kernel(dm0=dm0)
            if not self.method.converged:
                raise RuntimeError("SCF did not converge at a displaced geometry during the finite-difference Hessian.")
            return _asnumpy(self.build_gradient_method().kernel())

        try:
            return _central_difference_hessian(mol0, gradient_at, displacement)
        finally:
            self._reset_method(mol0.set_geom_(coords0, unit="Bohr", inplace=False))
            self._last_positions = None  # the SCF state belongs to a displaced geometry; rerun before reuse

    def _apply_solvent(self, mf, solvation: dict):
        solvation = rename_deprecated_keys(solvation, "driver.solvation")
        model = str(solvation.get("model", "ief-pcm"))
        if model.lower() == "cpcm":
            model = "C-PCM"
        if model.lower() in PCM_METHODS:
            mf = mf.PCM()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = model
            if "epsilon" in solvation:
                mf.with_solvent.eps = float(solvation["epsilon"])
            elif "solvent" in solvation:
                eps_table = read_pcm_eps()
                name = str(solvation["solvent"]).lower()
                if name not in eps_table:
                    raise ValueError(f"Unknown PCM solvent '{solvation['solvent']}'; give 'epsilon' explicitly.")
                mf.with_solvent.eps = eps_table[name]
            else:
                raise ValueError("For PCM solvation, specify either 'epsilon' or 'solvent' in the solvation block.")
        elif model.lower() == "smd":
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = "SMD"
            if "solvent_descriptors" in solvation:
                mf.with_solvent.solvent_descriptors = solvation["solvent_descriptors"]
            elif "solvent" in solvation:
                mf.with_solvent.solvent = solvation["solvent"]
            else:
                raise ValueError("For SMD solvation, specify either 'solvent_descriptors' or 'solvent'.")
        else:
            raise ValueError(f"Unknown solvation model '{model}'. Use one of {PCM_METHODS} or 'smd'.")
        return mf

    def build_method(self):
        if self.atoms is None:
            raise ValueError("Atoms are required to build the PySCF method.")
        config = dict(self.config)
        if self.is_3c:
            pyscf_xc, nlc, basis, ecp, _, _ = self._3c
            config.update(xc=pyscf_xc, nlc=nlc, basis=basis, ecp=ecp)
        mf = self._build_mf(config)
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import gen_disp_fun

            mf.get_dispersion = MethodType(gen_disp_fun(self._3c[4], self._3c[5]), mf)
            mf.do_disp = lambda: True
        return mf

    def build_gradient_method(self):
        # the cuEST wrapper overrides Gradients() to return its own gradient wrapper
        grad = self.method.Gradients() if self.accelerator == "cuest" else self.method.nuc_grad_method()
        if self.accelerator == "cuest":
            if not self.method.turn_on_cuest_hcore and hasattr(grad, "_hcore_energy"):
                grad.get_hcore_grad = MethodType(_native_hcore_gradient, grad)
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import gen_disp_grad_fun

            grad.get_dispersion = MethodType(gen_disp_grad_fun(self._3c[4], self._3c[5]), grad)
        return grad

    def build_hessian_method(self):
        method = self._native_method_for_hessian() if self.accelerator == "cuest" else self.method
        hess = method.Hessian()
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import gen_disp_hess_fun

            hess.get_dispersion = MethodType(gen_disp_hess_fun(self._3c[4], self._3c[5]), hess)
        hess.auxbasis_response = 2
        hess.grids_response = True
        return hess

    # ------------------------------------------------------------------ SCF
    def _reset_method(self, mol) -> None:
        """Reset cuEST and the native objects retained for disabled accelerator components."""
        if self.accelerator == "cuest":
            # CuESTWrapper.reset deliberately skips the native reset chain. Its backup objects otherwise
            # retain grids, DF integrals and PCM surfaces from the previous geometry.
            self.method.with_df.reset(mol)
            for grids in (self.method.grids, self.method.nlcgrids):
                grids._backup_grids.reset(mol)
            self.method._numint._backup_numint.reset()
            if getattr(self.method, "with_solvent", None) is not None:
                self.method.with_solvent._backup_with_solvent.reset(mol)
        self.method.reset(mol)

    def clear_cache(self) -> None:
        self._dm_cache = None
        self._native_method = None
        super().clear_cache()

    def run_kernel(self, atoms: Atoms | None = None, use_cache: bool = True) -> float:
        """Run the SCF at the current positions (skipped when already converged there); returns Eh.

        With ``use_cache`` the previous density matrix seeds the SCF. On non-convergence and
        ``retry_soscf`` a second-order (Newton) SCF is attempted.
        """
        self.update_atoms(atoms)
        if self.method is None:
            self.method = self.build_method()
        if not self._positions_changed() and self.method.converged:
            return self.method.e_tot

        mol = self.method.mol.set_geom_(self.atoms.get_positions(), unit="Angstrom", inplace=False)
        self._reset_method(mol)
        energy = self.method.kernel(dm0=self._dm_cache if use_cache else None)
        converged = self.method.converged
        if not converged and self.retry_soscf:
            newton = self.method.newton()
            newton.reset(mol)
            energy = newton.kernel()
            converged = newton.converged
            if converged:
                self.log("SOSCF converged")
                self.method.mo_coeff = newton.mo_coeff
                self.method.mo_occ = newton.mo_occ
                self.method.e_tot = newton.e_tot
                self.method.converged = True
        if converged:
            self._mark_positions()
        else:
            self._last_positions = None
        if use_cache:
            self._dm_cache = self.method.make_rdm1()
        return energy

    # ------------------------------------------------------------------ adapters
    def to_ase_calc(self) -> PySCFCalculator:
        return PySCFCalculator(self, max_unconverged_steps=self.config.get("max_unconverged_steps"))

    def to_pyscf_mf(self):
        self.run_kernel()  # no-op when already converged at the current positions
        return self.method

    # ------------------------------------------------------------------ implementations
    def _compute_energy_impl(self, atoms: Atoms | None = None) -> tuple[float, str]:
        e_tot = self.run_kernel(atoms)
        summary = self.method.scf_summary
        self.log(f"One-electron Energy [Eh]: {float(summary.get('e1', 0.0)):16.10f}")
        self.log(f"Coulomb Energy      [Eh]: {float(summary.get('coul', 0.0)):16.10f}")
        self.log(f"XC Energy           [Eh]: {float(summary.get('exc', 0.0)):16.10f}")
        for label, key in (("Dispersion Energy  ", "disp"), ("Solvent Energy     ", "solvent")):
            value = float(summary.get(key, 0.0))
            if abs(value) > 1e-10:
                self.log(f"{label} [Eh]: {value:16.10f}")
        self._log_frontier_orbitals()
        return e_tot, "Eh"

    def _log_frontier_orbitals(self) -> None:
        mo_energy = _asnumpy(self.method.mo_energy)
        na, nb = self.method.mol.nelec
        if mo_energy.ndim == 2:  # unrestricted
            alpha, beta = np.sort(mo_energy[0]), np.sort(mo_energy[1])
            if na < alpha.size:
                self.log(f"LUMO Alpha [Eh]: {alpha[na]:12.6f}")
            if nb < beta.size:
                self.log(f"LUMO Beta  [Eh]: {beta[nb]:12.6f}")
            self.log(f"HOMO Alpha [Eh]: {alpha[na - 1]:12.6f}")
            self.log(f"HOMO Beta  [Eh]: {beta[nb - 1]:12.6f}")
        else:  # restricted or restricted open-shell
            energies = np.sort(mo_energy)
            if na < energies.size:
                self.log(f"LUMO [Eh]: {energies[na]:12.6f}")
            self.log(f"HOMO [Eh]: {energies[na - 1]:12.6f}")

    def _compute_forces_impl(self, atoms: Atoms | None = None) -> tuple[np.ndarray, str]:
        self.run_kernel(atoms)
        gradient = _asnumpy(self.build_gradient_method().kernel())
        return -gradient, "Eh/Bohr"

    def _compute_hessian_impl(self, atoms: Atoms | None) -> np.ndarray:
        self.run_kernel(atoms)
        numerical = self.config.get("numerical_hessian")
        if numerical is None:
            numerical = self.accelerator == "cuest"  # cuEST has no analytic Hessian
        if numerical:
            step = float(self.config.get("finite_diff_step", 5e-3)) / Bohr  # Angstrom -> Bohr
            if self.accelerator == "cuest":
                self.log("cuEST: Hessian from central differences of the accelerated gradients.")
                return self._finite_difference_hessian_cuest(step)
            try:
                return _finite_difference_hessian(self.build_gradient_method(), step)
            finally:
                self.method.reset(self.method.mol)  # the scanner re-pointed the shared grids / DF objects
        if self.accelerator == "cuest":
            self.log("cuEST: analytic Hessian from a native gpu4pyscf method at this geometry.")
        return _asnumpy(self.build_hessian_method().kernel())
