import os
import sys
from types import MethodType
from typing import Optional, Tuple

import numpy as np
from pyscf import gto, lib, dft
from pyscf.lib import GradScanner
from pyscf.gto import charge, Mole
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import Calculator, all_changes

from .base_driver import BaseDriver
from qc_launcher.utils.utils import read_pcm_eps


class PySCFCalculator(Calculator):
    """
    PySCF calculator for ASE.
    This calculator uses PySCF to compute the energy and forces of a system.
    It can be used with various mean field methods provided by PySCF.
    """
    implemented_properties = ["energy", "forces"]
    default_parameters = {}
    def __init__(
        self,
        method,
        g_scanner,
        max_unconverged_steps: int = None,
        retry_soscf=False,
        **kwargs,
    ):
        self.method = method
        self.g_scanner: GradScanner = g_scanner
        self.retry_soscf = retry_soscf
        self.max_unconverged_steps = sys.maxsize if max_unconverged_steps is None else max_unconverged_steps
        self.num_unconverged = 0
        Calculator.__init__(self, **kwargs)

    def set_max_unconverged_steps(self, tol: int = None):
        self.max_unconverged_steps = sys.maxsize if tol is None else tol
        self.num_unconverged = 0
    
    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(
        self,
        atoms: Atoms = None,
        properties=None, 
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        mol: Mole = self.method.mol
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([charge(x) for x in mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))
        
        mol.set_geom_(_atoms, unit="Angstrom")
        
        energy, gradients = self.g_scanner(mol)
        if not self.g_scanner.converged and self.retry_soscf:
            # try SOSCF if not converged
            newton_method = self.method.newton()
            newton_method.reset(mol)
            newton_method.kernel()
            self.g_scanner.base.mo_coeff = newton_method.mo_coeff
            self.g_scanner.base.mo_occ = newton_method.mo_occ
            energy, gradients = self.g_scanner(mol)
            if self.g_scanner.converged:
                print("SOSCF converged")
        if not self.g_scanner.converged:
            self.num_unconverged += 1
            if self.num_unconverged > self.max_unconverged_steps:
                print(f"Warning: SCF has not converged for {self.num_unconverged} steps")
                # raise RuntimeError(f"SCF failed to converge after {self.num_unconverged} steps.")

        # store the energy and forces
        self.results["energy"] = energy * Hartree
        self.results["forces"] = -gradients * (Hartree / Bohr)


class PySCFDriver(BaseDriver):
    def __init__(
        self,
        atoms: Atoms,
        config: dict,
        **kwargs,
    ):
        super().__init__(atoms=atoms, config=config, **kwargs)
        xc: str = self.config.get("xc", "B3LYP")
        self.is_3c = xc.lower().endswith("3c")
        self.method = None if atoms is None else self.build_method(atoms=atoms)
        self._dm_cache = None  # cache for density matrix
        self.retry_soscf = self.config.get("retry_soscf", False)
    
    def _build_mf(self, atoms: Optional[Atoms] = None, config: Optional[dict] = None):
        """
        Build PySCF method from configuration or atoms.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            PySCF SCF method object.
        """
        self.update_atoms(atoms)
        config = self.config if config is None else config
        xc: str = config.get("xc", "B3LYP")
        restricted: Optional[bool] = config.get("restricted", None)
        basis: str = config.get("basis", "def2-SVP")
        ecp: Optional[str] = config.get("ecp", None)
        nlc: str = config.get("nlc", '')
        disp: Optional[str] = config.get("disp", None)
        grids: Optional[dict] = config.get("grids", None)
        nlcgrids: Optional[dict] = config.get("nlcgrids", None)
        verbose: int = config.get("verbose", 2)
        scf_conv_tol: float = float(config.get("scf_conv_tol", 1e-8))
        direct_scf_tol: float = float(config.get("direct_scf_tol", 1e-8))
        scf_max_cycle: int = int(config.get("scf_max_cycle", 50))
        level_shift: float = float(config.get("level_shift", 0.0))
        diis_space: int = int(config.get("diis_space", 8))
        with_df: bool = config.get("with_df", True)
        auxbasis: Optional[str] = config.get("auxbasis", None)
        with_gpu: bool = config.get("with_gpu", True)
        with_solvent = config.get("with_solvent", False)
        solvent = config.get("solvent", {"method": "ief-pcm", "eps": 78.3553, "solvent": "water"})

        max_memory = config.get("max_memory", None)
        if max_memory is not None:
            max_memory *= 1024  # convert GB to MB
        threads = config.get("threads", os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
        lib.num_threads(threads)

        atom = [(symbol, pos) for symbol, pos in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())]
        charge = self.atoms.info.get("charge", 0)
        spin = self.atoms.info.get("multiplicity", 1) - 1

        # build molecule
        mol = gto.M(
            atom=atom,
            basis=basis,
            ecp=ecp,
            max_memory=max_memory,
            verbose=verbose,
            charge=charge,
            spin=spin,
        )
        mol.build()

        # build Kohn-Sham object
        if restricted is True:
            mf = dft.RKS(mol, xc=xc) if spin % 2 == 0 else dft.ROKS(mol, xc=xc)
        elif restricted is False:
            mf = dft.UKS(mol, xc=xc)
        else:
            mf = dft.KS(mol, xc=xc)
        mf.nlc = nlc
        mf.disp = disp
        # set grids
        if grids is not None:
            if "atom_grid" in grids:
                mf.grids.atom_grid = grids["atom_grid"]
            elif "level" in grids:
                mf.grids.level = grids["level"]
        # set nlc grids
        if mf._numint.libxc.is_nlc(mf.xc) or nlc is not None:
            if nlcgrids is not None:
                if "atom_grid" in nlcgrids:
                    mf.nlcgrids.atom_grid = nlcgrids["atom_grid"]
                if "level" in nlcgrids:
                    mf.nlcgrids.level = nlcgrids["level"]
        # set density fitting
        if with_df:
            mf = mf.density_fit(auxbasis=auxbasis)

        # move to GPU if requested
        if with_gpu:
            try:
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
                mf = mf.to_gpu()
            except ImportError:
                self.log("GPU support is not available. Proceeding with CPU.")
    
        # solvation model
        if with_solvent:
            solvent = solvent
            if solvent["method"].lower() in ["c-pcm", "ief-pcm", "ss(v)pe", "cosmo"]:
                mf = mf.PCM()
                mf.with_solvent.lebedev_order = 29
                mf.with_solvent.method = solvent["method"]
                if "eps" in solvent:
                    mf.with_solvent.eps = solvent["eps"]
                elif "solvent" in solvent:
                    eps_dict = read_pcm_eps()
                    assert solvent["solvent"].lower() in eps_dict, f"Unknown solvent: {solvent['solvent']}"
                    mf.with_solvent.eps = eps_dict[solvent["solvent"].lower()]
                else:
                    raise ValueError("For PCM solvent, either 'eps' or 'solvent' must be specified in config")
            elif solvent["method"].lower() == "smd":
                mf = mf.SMD()
                mf.with_solvent.lebedev_order = 29
                mf.with_solvent.method = "SMD"
                if "solvent_descriptor" in solvent:
                    mf.with_solvent.solvent_descriptor = solvent["solvent_descriptor"]
                elif "solvent" in solvent:
                    mf.with_solvent.solvent = solvent["solvent"]
                else:
                    raise ValueError("For SMD solvent, either 'solvent_descriptor' or 'solvent' must be specified in config")
            else:
                raise ValueError(f"Unknown solvent method: {solvent['method']}")

        mf.direct_scf_tol = direct_scf_tol
        mf.chkfile = None  # disable checkpoint file generation
        mf.conv_tol = scf_conv_tol
        mf.max_cycle = scf_max_cycle
        mf.level_shift = level_shift
        mf.diis_space = diis_space

        return mf

    def build_method(self, atoms: Optional[Atoms] = None):
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_fun
            xc = self.config.get("xc", "B973c").lower()
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            # modify config dictionary for 3c method
            config_3c = self.config.copy()
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc)
            config_3c["xc"] = pyscf_xc
            config_3c["nlc"] = nlc
            config_3c["basis"] = basis
            config_3c["ecp"] = ecp
            # build method with 3c config
            mf = self._build_mf(atoms=atoms, config=config_3c)
            # attach 3c specific attributes
            mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
            mf.do_disp = lambda: True
            return mf
        return self._build_mf(atoms=atoms)

    def build_gradient_method(self):
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_grad_fun
            xc = self.config.get("xc", "B973c").lower()
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc)
            g = self.method.nuc_grad_method()
            g.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), g)
            return g
        return self.method.nuc_grad_method()

    def build_hessian_method(self):
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_hess_fun
            xc = self.config.get("xc", "B973c").lower()
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc)
            h = self.method.Hessian()
            h.get_dispersion = MethodType(gen_disp_hess_fun(xc_disp, xc_gcp), h)
            h.auxbasis_response = 2
        else:
            h = self.method.Hessian()
        h.auxbasis_response = 2
        h.grids_response = True
        return h

    def clear_cache(self):
        self._dm_cache = None
        return super().clear_cache()

    def run_kernel(self, atoms: Optional[Atoms] = None, use_cache: bool = True) -> float:
        """
        run PySCF kernel to compute energy, with optional caching of density matrix for faster convergence in subsequent calls.
        """
        updated = self.update_atoms(atoms)
        # lazy build method
        if self.method is None:
            self.method = self.build_method()
        
        if not updated and self.method.converged:
            # positions are the same and SCF already converged, no need to rerun
            return self.method.e_tot
        
        # update molecule geometry
        mol = self.method.mol.set_geom_(self.atoms.get_positions(), unit="Angstrom", inplace=False)
        self.method.reset(mol)  # reset method with new molecule geometry
        # if use_cache, pass cached density matrix to speed up convergence
        dm0 = self._dm_cache if use_cache else None
        energy = self.method.kernel(dm0=dm0)
        converged = self.method.converged
        if not converged and self.retry_soscf:
            # try SOSCF if not converged
            newton_method = self.method.newton()
            newton_method.reset(mol)
            energy = newton_method.kernel()
            converged = newton_method.converged
            if converged:
                self.log("SOSCF converged")
                self.method.mo_coeff = newton_method.mo_coeff
                self.method.mo_occ = newton_method.mo_occ
        if use_cache:  # save new density matrix to cache
            self._dm_cache = self.method.make_rdm1()
        
        return energy  # in Hartree

    def to_ase_calc(self):
        gradient_method = self.build_gradient_method()
        g_scanner = gradient_method.as_scanner()
        return PySCFCalculator(
            method=self.method, g_scanner=g_scanner,
            max_unconverged_steps=self.config.get("max_unconverged_steps", None),
            retry_soscf=self.retry_soscf,
        )

    def to_pyscf_mf(self):
        if self.method is None:
            self.method = self.build_method()
        if not self.method.converged:
            self.run_kernel(use_cache=False)  # run SCF to ensure method is converged before returning
        return self.method

    def _compute_energy_impl(self, atoms: Optional[Atoms] = None) -> Tuple[float, str]:
        e_tot = self.run_kernel(atoms=atoms, use_cache=True)
        scf_summary = self.method.scf_summary
        e1 = scf_summary.get("e1", 0.0)        # one-electron energy
        e_coul = scf_summary.get("coul", 0.0)  # Coulomb energy
        e_xc = scf_summary.get("exc", 0.0)     # exchange-correlation energy
        e_disp = scf_summary.get("disp", 0.0)  # dispersion energy
        e_solvent = scf_summary.get("solvent", 0.0)  # solvent energy
        # log results
        self.log(f"One-electron Energy [Eh]: {e1:16.10f}")
        self.log(f"Coulomb Energy      [Eh]: {e_coul:16.10f}")
        self.log(f"XC Energy           [Eh]: {e_xc:16.10f}")
        if abs(e_disp) > 1e-10:
            self.log(f"Dispersion Energy   [Eh]: {e_disp:16.10f}")
        if abs(e_solvent) > 1e-10:
            self.log(f"Solvent Energy      [Eh]: {e_solvent:16.10f}")
        dm = self.method.make_rdm1()
        if not isinstance(dm, np.ndarray):
            dm = dm.get()  # convert cupy array to numpy array if needed
        mo_energy = self.method.mo_energy
        if not isinstance(mo_energy, np.ndarray):
            mo_energy = mo_energy.get()  # convert cupy array to numpy array if needed
        if dm.ndim == 3:  # open-shell
            mo_energy[0].sort()
            mo_energy[1].sort()
            na, nb = self.method.nelec
            self.log(f"LUMO Alpha [Eh]: {mo_energy[0][na]:12.6f}")
            self.log(f"LUMO Beta  [Eh]: {mo_energy[1][nb]:12.6f}")
            self.log(f"HOMO Alpha [Eh]: {mo_energy[0][na-1]:12.6f}")
            self.log(f"HOMO Beta  [Eh]: {mo_energy[1][nb-1]:12.6f}")
        else:  # closed-shell
            mo_energy.sort()
            nocc = self.method.mol.nelectron // 2
            self.log(f"LUMO [Eh]: {mo_energy[nocc]:12.6f}")
            self.log(f"HOMO [Eh]: {mo_energy[nocc-1]:12.6f}")
        return e_tot, "Eh"

    def _compute_forces_impl(self, atoms: Optional[Atoms]) -> Tuple[np.ndarray, str]:
        self.run_kernel(atoms=atoms, use_cache=True)
        gradient_method = self.build_gradient_method()
        grad = gradient_method.kernel()
        return -grad, "Eh/Bohr"

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
    ) -> np.ndarray:
        numerical_hess = self.config.get("numerical_hess", False)
        if numerical_hess:
            with_gpu = self.config.get("with_gpu", True)
            if with_gpu:
                from qc_launcher.utils import finite_diff_gpu as finite_diff
            else:
                from pyscf.tools import finite_diff
            finite_diff_eps = self.config.get("finite_diff_eps", 5e-3)
            gradient_method = self.build_gradient_method()
            finite_diff_h = finite_diff.Hessian(gradient_method)
            finite_diff_h.displacement = finite_diff_eps / Bohr  # convert from Angstrom to Bohr
            hessian = finite_diff_h.kernel()
        else:
            self.run_kernel(atoms=atoms, use_cache=True)
            hessian_method = self.build_hessian_method()
            hessian = hessian_method.kernel()
        
        return hessian
    
    def compute_resp(self, atoms: Optional[Atoms] = None) -> np.ndarray:
        from gpu4pyscf.pop import esp
        from qc_launcher.utils.topology import get_constraints_idx, rdkit_mol_from_pyscf
        self.run_kernel(atoms=atoms, use_cache=True)
        dm = self.method.make_rdm1()
        # stage 1: RESP fitting under weak hyperbolic penalty
        q1 = esp.resp_solve(self.method.mol, dm)
        # stage 2: RESP fitting with constraints
        rdkit_mol = rdkit_mol_from_pyscf(self.method.mol)
        sum_constraints_idx, equal_constraints = get_constraints_idx(rdkit_mol)
        sum_constraints = []
        for i in sum_constraints_idx:
            sum_constraints.append([q1[i], [i]])
        q2 = esp.resp_solve(
            self.method.mol, dm,
            resp_a=1e-3,
            sum_constraints=sum_constraints,
            equal_constraints=equal_constraints,
        )
        # print RESP charges
        self.log("RESP charges [e]:")
        for i, charge in enumerate(q2):
            self.log(f"{i+1:3d} {charge:16.10f}")
        return q2

