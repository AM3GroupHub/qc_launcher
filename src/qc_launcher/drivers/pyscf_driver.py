"""PySCF / gpu4pyscf driver: Kohn-Sham DFT energies, gradients and Hessians, optionally on the GPU."""

from __future__ import annotations

import os
import warnings
from types import MethodType

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree
from pyscf import dft, gto, lib

from qc_launcher.utils.config import ConfigDeprecationWarning, rename_deprecated_keys
from qc_launcher.utils.utils import read_pcm_eps

from .base_driver import BaseDriver

PCM_METHODS = ("c-pcm", "ief-pcm", "ss(v)pe", "cosmo")


def _asnumpy(array) -> np.ndarray:
    """Return a NumPy array for either a NumPy or a CuPy input."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _parse_3c(xc: str) -> tuple[str, str, str, str | None, str, str]:
    """Split a composite '3c' functional (e.g. B97-3c) into (xc, nlc, basis, ecp, xc_disp, xc_gcp)."""
    from gpu4pyscf.drivers.dft_3c_driver import parse_3c

    pyscf_xc, nlc, basis, ecp, (xc_disp, _), xc_gcp = parse_3c(xc.lower().replace("-3c", "3c"))
    return pyscf_xc, nlc, basis, ecp, xc_disp, xc_gcp


def _finite_difference_hessian(grad_method, displacement: float) -> np.ndarray:
    """Central-difference Hessian, (N, N, 3, 3) in Eh/Bohr^2, from a pyscf or gpu4pyscf gradient method.

    ``displacement`` is in Bohr. The molecule is displaced in place and restored afterwards; the caller
    should reset its mean-field object because the scanner re-points the shared grids / DF objects.
    """
    scanner = grad_method.as_scanner()
    mol = grad_method.mol
    coords0 = mol.atom_coords(unit="Bohr").copy()
    natm = mol.natm
    hessian = np.zeros((natm, natm, 3, 3))

    def gradient_at(coords):
        mol.set_geom_(coords, unit="Bohr")
        _, gradient = scanner(mol)
        if not scanner.converged:
            raise RuntimeError("SCF did not converge at a displaced geometry during the finite-difference Hessian.")
        return _asnumpy(gradient)

    try:
        for i in range(natm):
            for j in range(3):
                coords = coords0.copy()
                coords[i, j] += displacement
                grad_plus = gradient_at(coords)
                coords[i, j] -= 2.0 * displacement
                grad_minus = gradient_at(coords)
                hessian[i, :, j, :] = (grad_plus - grad_minus) / (2.0 * displacement)
    finally:
        mol.set_geom_(coords0, unit="Bohr")
    return 0.5 * (hessian + hessian.transpose(1, 0, 3, 2))


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
    ``max_unconverged_steps``, ``numerical_hessian`` (False), ``finite_diff_step`` (5e-3 Angstrom).
    """

    def __init__(self, atoms: Atoms | None, config: dict, **kwargs):
        super().__init__(atoms=atoms, config=config, **kwargs)
        xc = str(self.config.get("xc", "B3LYP"))
        self.is_3c = xc.lower().endswith("3c")
        self._3c = _parse_3c(xc) if self.is_3c else None
        self.retry_soscf = bool(self.config.get("retry_soscf", False))
        self._dm_cache = None
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
        return mf

    def _apply_solvent(self, mf, solvation: dict):
        solvation = rename_deprecated_keys(solvation, "driver.solvation")
        model = str(solvation.get("model", "ief-pcm"))
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
        grad = self.method.nuc_grad_method()
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import gen_disp_grad_fun

            grad.get_dispersion = MethodType(gen_disp_grad_fun(self._3c[4], self._3c[5]), grad)
        return grad

    def build_hessian_method(self):
        hess = self.method.Hessian()
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import gen_disp_hess_fun

            hess.get_dispersion = MethodType(gen_disp_hess_fun(self._3c[4], self._3c[5]), hess)
        hess.auxbasis_response = 2
        hess.grids_response = True
        return hess

    # ------------------------------------------------------------------ SCF
    def clear_cache(self) -> None:
        self._dm_cache = None
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
        self.method.reset(mol)
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
        if self.config.get("numerical_hessian", False):
            step = float(self.config.get("finite_diff_step", 5e-3)) / Bohr  # Angstrom -> Bohr
            try:
                return _finite_difference_hessian(self.build_gradient_method(), step)
            finally:
                self.method.reset(self.method.mol)  # the scanner re-pointed the shared grids / DF objects
        return _asnumpy(self.build_hessian_method().kernel())
