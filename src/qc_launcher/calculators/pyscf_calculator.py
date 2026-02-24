"""
PySCF calculator implementation.
"""

import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np
from pyscf import gto, scf
from pyscf.lib import GradScanner
from pyscf.gto import charge, Mole
from ase import Atoms, units

from qc_launcher.common.base_calculator import BaseQCCalculator
from qc_launcher.common.utils import build_method, build_3c_method, get_gradient_method, get_Hessian_method


class PySCFCalculator(BaseQCCalculator):
    """
    PySCF calculator implementation.
    
    This calculator uses PySCF to compute energy, forces, and Hessian.
    Supports various DFT functionals and basis sets.
    
    Example:
        >>> config = {
        ...     'xc': 'B3LYP',
        ...     'basis': 'def2-SVP',
        ...     'charge': 0,
        ...     'spin': 0,
        ... }
        >>> calc = PySCFCalculator(config=config)
        >>> atoms.calc = calc
        >>> energy = atoms.get_potential_energy()
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        method: Optional[scf.hf.SCF] = None,
        xc_3c: Optional[str] = None,
        soscf: bool = False,
        max_unconverged_steps: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize PySCF calculator.
        
        Args:
            config: Configuration dictionary.
            method: Pre-built PySCF method (if None, will be built from config).
            xc_3c: 3C correction functional name.
            soscf: Whether to use second-order SCF for convergence.
            max_unconverged_steps: Maximum number of unconverged steps allowed.
            **kwargs: Additional arguments for base calculator.
        """
        super().__init__(config=config, **kwargs)
        
        self.method = method
        self.xc_3c = xc_3c
        self.soscf = soscf
        self.max_unconverged_steps = sys.maxsize if max_unconverged_steps is None else max_unconverged_steps
        self.num_unconverged = 0
        self.g_scanner: Optional[GradScanner] = None
        
    def _build_method(self, atoms: Atoms) -> scf.hf.SCF:
        """
        Build PySCF method from configuration or atoms.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            PySCF SCF method object.
        """
        if self.method is not None:
            return self.method
            
        # Update config with atoms info
        if "charge" in atoms.info:
            self.config["charge"] = atoms.info["charge"]
        if "multiplicity" in atoms.info:
            self.config["spin"] = atoms.info["multiplicity"] - 1
            
        # Build method based on functional type
        xc = self.config.get("xc", "B3LYP")
        if xc.lower().endswith("3c"):
            self.xc_3c = xc
            method = build_3c_method(self.config)
        else:
            method = build_method(self.config)
            
        # Build gradient scanner
        self.g_scanner = get_gradient_method(method, xc_3c=self.xc_3c).as_scanner()
        
        return method
    
    def _compute_energy_and_forces(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces using PySCF.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Tuple of (energy in eV, forces in eV/Angstrom).
        """
        if self.g_scanner is None:
            self.g_scanner = get_gradient_method(self.method, xc_3c=self.xc_3c).as_scanner()
            
        mol: Mole = self.method.mol
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        
        # Check if atomic numbers match
        Z = np.array([charge(x) for x in mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))
        
        mol.set_geom_(_atoms, unit="Angstrom")
        
        # Compute energy and gradients
        energy, gradients = self.g_scanner(mol)
        
        # Try SOSCF if not converged
        if not self.g_scanner.converged and self.soscf:
            newton_method = self.method.newton()
            newton_method.reset(mol)
            newton_method.kernel()
            self.g_scanner.base.mo_coeff = newton_method.mo_coeff
            self.g_scanner.base.mo_occ = newton_method.mo_occ
            energy, gradients = self.g_scanner(mol)
            if self.g_scanner.converged:
                print("SOSCF converged")
        
        # Check convergence
        if not self.g_scanner.converged:
            self.num_unconverged += 1
            if self.num_unconverged > self.max_unconverged_steps:
                raise RuntimeError(f"SCF failed to converge after {self.num_unconverged} steps.")
        
        self._converged = self.g_scanner.converged
        
        # Convert units
        energy_ev = energy * units.Hartree
        forces = -gradients * (units.Hartree / units.Bohr)
        
        return energy_ev, forces
    
    def _compute_hessian_impl(self, atoms: Atoms) -> np.ndarray:
        """
        Compute Hessian matrix using PySCF.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Hessian matrix in eV/Angstrom^2 with shape (3N, 3N).
        """
        # Update geometry
        self.method.mol.set_geom_(atoms.get_positions(), unit="Angstrom")
        self.method.run()
        
        # Compute Hessian
        hessian = get_Hessian_method(self.method, xc_3c=self.xc_3c).kernel()
        natom = self.method.mol.natm
        hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
        
        # Convert units from Hartree/Bohr^2 to eV/Angstrom^2
        hessian *= (units.Hartree / units.Bohr**2)
        
        return hessian
    
    def set_max_unconverged_steps(self, tol: Optional[int] = None):
        """
        Set maximum number of unconverged steps.
        
        Args:
            tol: Maximum steps (None for unlimited).
        """
        self.max_unconverged_steps = sys.maxsize if tol is None else tol
        self.num_unconverged = 0
