import os
import sys
from types import MethodType
from typing import Optional, Literal

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
    ):
        super().__init__(atoms=atoms, config=config)
        self.is_3c = self.config.get("xc", "B3LYP").endswith("3c")
        self.method = self.build_method()
        self.gradient_method = None
        self.hessian_method = None
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
        xc = config.get("xc", "B3LYP")
        basis = config.get("basis", "def2-svp")
        ecp = config.get("ecp", None)
        nlc = config.get("nlc", None)
        disp = config.get("disp", None)
        grids = config.get("grids", None)
        nlcgrids = config.get("nlcgrids", None)
        verbose = config.get("verbose", 2)
        scf_conv_tol = float(config.get("scf_conv_tol", 1e-8))
        direct_scf_tol = float(config.get("direct_scf_tol", 1e-8))
        scf_max_cycle = int(config.get("scf_max_cycle", 50))
        level_shift = float(config.get("level_shift", 0.0))
        diis_space = int(config.get("diis_space", 8))
        with_df = config.get("with_df", True)
        auxbasis = config.get("auxbasis", None)
        with_gpu = config.get("with_gpu", True)

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
                print("GPU support is not available. Proceeding with CPU.")
    
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
            xc = self.config.get("xc", "B973c")
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            # modify config dictionary for 3c method
            config_3c = self.config.copy()
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc.lower())
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
            xc = self.config.get("xc", "B973c")
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc.lower())
            g = self.method.nuc_grad_method()
            g.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), g)
            return g
        return self.method.nuc_grad_method()

    def build_hessian_method(self):
        if self.is_3c:
            from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_hess_fun
            xc = self.config.get("xc", "B973c")
            xc = xc.replace("-3c", "3c")  # allow both "B973c" and "B97-3c"
            _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc.lower())
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
        self.update_atoms(atoms)
        self.method.mol.set_geom_(self.atoms.get_positions(), unit="Angstrom")
        dm0 = self._dm_cache if use_cache else None
        energy = self.method.kernel(dm0=dm0)
        converged = self.method.converged
        if not converged and self.retry_soscf:
            # try SOSCF if not converged
            mo_init = self.method.mo_coeff
            mocc_init = self.method.mo_occ
            newton_method = self.method.newton()
            energy = newton_method.kernel(mo_init, mocc_init)
            converged = newton_method.converged
            if converged:
                print("SOSCF converged")
                self.method.mo_coeff = newton_method.mo_coeff
                self.method.mo_occ = newton_method.mo_occ

        if use_cache:
            self._dm_cache = self.method.make_rdm1()
        return energy * Hartree

    def to_ase_calc(self):
        if self.gradient_method is None:
            self.gradient_method = self.build_gradient_method()
        g_scanner = self.gradient_method.as_scanner()
        return PySCFCalculator(
            method=self.method, g_scanner=g_scanner,
            max_unconverged_steps=self.config.get("max_unconverged_steps", None),
            retry_soscf=self.retry_soscf,
        )

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
        hess_format: Literal["pyscf", "ase"] = "ase",
    ) -> np.ndarray:
        # run energy
        self.run_kernel(atoms=atoms, use_cache=True)
        
        # Compute Hessian
        if self.hessian_method is None:
            self.hessian_method = self.build_hessian_method()
        hessian = self.hessian_method.kernel()
        hessian = self._convert_hessian_format(hessian=hessian, hess_format=hess_format)
        return hessian
    