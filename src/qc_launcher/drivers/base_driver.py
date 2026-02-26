from abc import ABC, abstractmethod
from typing import Optional, Literal

import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.calculator import Calculator


class BaseDriver(ABC):    
    def __init__(self, atoms: Atoms, config: dict):
        """
        Initialize the base driver.
        
        Args:
            atoms: ASE Atoms object.
            config: Configuration dictionary containing driver parameters.
            **kwargs: Additional keyword arguments for ASE Calculator.
        """
        self.atoms = atoms
        self.config = config
        self._hessian_cache = {}
            
    def update_atoms(self, atoms: Atoms) -> None:
        """
        Update the internal state with new atomic positions.
        
        This is software-specific and must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object with updated positions.
        """
        # check if atomic numbers have changed
        new_numbers = atoms.get_atomic_numbers()
        curr_numbers = self.atoms.get_atomic_numbers()
        if not np.array_equal(new_numbers, curr_numbers):
            raise ValueError("Atomic numbers cannot be changed. Please create a new driver instance.")
        new_charge = atoms.info.get("charge", 0)
        curr_charge = self.atoms.info.get("charge", 0)
        if new_charge != curr_charge:
            raise ValueError("Charge cannot be changed. Please create a new driver instance.")
        new_multiplicity = atoms.info.get("multiplicity", 1)
        curr_multiplicity = self.atoms.info.get("multiplicity", 1)
        if new_multiplicity != curr_multiplicity:
            raise ValueError("Multiplicity cannot be changed. Please create a new driver instance.")
        self.atoms = atoms

    @abstractmethod
    def to_ase_calc(self) -> Calculator:
        """
        Convert the internal method/model to an ASE Calculator if possible.
        
        This is software-specific and must be implemented by each subclass.
        
        Returns:
            An ASE Calculator object if conversion is possible, otherwise None.
        """
        pass


    @abstractmethod
    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
        hess_format: Literal["pyscf", "ase"] = "ase",
    ) -> np.ndarray:
        """
        Compute the Hessian matrix.
        
        This is software-specific and must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Hessian matrix, pyscf format (N, N, 3, 3) in a.u., or ase format (3N, 3N) in eV/Angstrom^2.
        """
        pass
    
    def compute_hessian(
        self, 
        atoms: Optional[Atoms] = None,
        use_cache: bool = True,
        hess_format: Literal["pyscf", "ase"] = "ase",
    ) -> np.ndarray:
        """
        Compute the Hessian matrix with optional caching.
        
        Args:
            atoms: ASE Atoms object.
            use_cache: Whether to use cached Hessian if positions haven't changed.
            hess_format: Desired format of the Hessian ('pyscf' or 'ase').
            
        Returns:
            Hessian matrix in the specified format.
        """
        if hess_format not in {"pyscf", "ase"}:
            raise ValueError(
                "Invalid hess_format. Use 'pyscf' or 'ase'."
            )
        atoms = self.atoms if atoms is None else atoms
        positions = atoms.get_positions()
        cache_key = self._get_cache_key(positions)
        
        # Check cache
        if use_cache and cache_key in self._hessian_cache:
            cached = self._hessian_cache[cache_key]
            return self._convert_hessian_format(atoms, cached, hess_format)
        
        # Compute Hessian
        hessian = self._compute_hessian_impl(atoms, hess_format=hess_format)
        
        # Cache the result
        if use_cache:
            self._hessian_cache[cache_key] = hessian
            
        return hessian
        
    def clear_cache(self):
        """Clear all cached data."""
        self._hessian_cache.clear()
            
        # ==================== Helper Methods ====================
    
    def _get_cache_key(self, positions: np.ndarray, tolerance: float = 1e-4) -> str:
        """
        Generate a cache key for the given positions.
        
        Args:
            positions: Atomic positions.
            tolerance: Tolerance for position comparison.
            
        Returns:
            Cache key string.
        """
        # Round positions to avoid floating point comparison issues
        rounded = np.round(positions / tolerance).astype(int)
        return str(hash(rounded.tobytes()))

    def _convert_hessian_format(
        self,
        hessian: np.ndarray,
        hess_format: Literal["pyscf", "ase"],
    ) -> np.ndarray:
        natm = len(self.atoms)
        if hess_format == "pyscf":
            if hessian.shape == (natm, natm, 3, 3):
                return hessian
            elif hessian.shape == (3*natm, 3*natm):
                hessian = hessian.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
                return hessian * (Bohr**2 / Hartree)  # convert to a.u.
            else:
                raise ValueError("Invalid Hessian shape")
        elif hess_format == "ase":
            if hessian.shape == (3*natm, 3*natm):
                return hessian
            elif hessian.shape == (natm, natm, 3, 3):
                hessian = hessian.transpose(0, 2, 1, 3).reshape(3*natm, 3*natm)
                return hessian * (Hartree / Bohr**2)  # convert to eV/Angstrom^2
            else:
                raise ValueError("Invalid Hessian shape")

        raise ValueError("Invalid hess_format. Use 'pyscf' or 'ase'.")
    