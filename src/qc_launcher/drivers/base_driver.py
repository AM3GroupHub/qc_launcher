from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import numpy as np
from ase import Atoms, units


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
        
    @abstractmethod
    def build_method(self, atoms: Optional[Atoms] = None) -> Any:
        """
        Build the underlying method/model object.
        
        This is software-specific and must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            The method/model object specific to the software.
        """
        pass
    
    @abstractmethod
    def _compute_hessian_impl(self, atoms: Optional[Atoms]) -> np.ndarray:
        """
        Compute the Hessian matrix.
        
        This is software-specific and must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Hessian matrix with shape (3N, 3N) in eV/Angstrom^2.
        """
        pass
    
    def compute_hessian(
        self, 
        atoms: Optional[Atoms] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Compute the Hessian matrix with optional caching.
        
        Args:
            atoms: ASE Atoms object.
            use_cache: Whether to use cached Hessian if positions haven't changed.
            
        Returns:
            Hessian matrix with shape (3N, 3N) in eV/Angstrom^2.
        """
        if atoms is None:
            atoms = self.atoms
        positions = atoms.get_positions()
        cache_key = self._get_cache_key(positions)
        
        # Check cache
        if use_cache and cache_key in self._hessian_cache:
            return self._hessian_cache[cache_key]
        
        # Compute Hessian
        hessian = self._compute_hessian_impl(atoms)
        
        # Cache the result
        if use_cache:
            self._hessian_cache[cache_key] = hessian
            
        return hessian
    
    def get_vibrational_modes(
        self,
        atoms: Optional[Atoms] = None,
        hessian: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute vibrational frequencies and normal modes from Hessian.
        
        Args:
            atoms: ASE Atoms object.
            hessian: Pre-computed Hessian matrix (will compute if None).
            
        Returns:
            Dictionary containing:
                - 'frequencies': Frequencies in cm^-1 (imaginary freqs are negative)
                - 'modes': Normal modes with shape (3N, 3N)
                - 'reduced_mass': Reduced masses
        """
        if hessian is None:
            hessian = self.compute_hessian(atoms)
            
        # Mass-weighted Hessian
        masses = atoms.get_masses()
        mass_weights = np.repeat(masses, 3)
        mw_hessian = hessian / np.sqrt(mass_weights[:, None] * mass_weights[None, :])
        
        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(mw_hessian)
        
        # Convert eigenvalues to frequencies
        # frequency = sqrt(eigenvalue) * conversion_factor
        conversion = 1e-3 * units._hbar / (units._e * units._amu) * 1e20  # to cm^-1
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion / (2 * np.pi * units._c * 100)
        
        # Compute reduced masses
        reduced_mass = 1.0 / np.sum(eigenvectors**2, axis=0)
        
        return {
            'frequencies': frequencies,
            'modes': eigenvectors,
            'reduced_mass': reduced_mass,
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self._hessian_cache.clear()
            
        # ==================== Helper Methods ====================
    
    def _get_cache_key(self, positions: np.ndarray, tolerance: float = 1e-8) -> str:
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

