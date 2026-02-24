"""
Base calculator class for quantum chemistry calculations.

This module provides an abstract base class for different quantum chemistry software
(PySCF, MACE, UMA, etc.) to inherit from, ensuring a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes


class BaseQCCalculator(Calculator, ABC):
    """
    Abstract base calculator for quantum chemistry calculations.
    
    This class provides a unified interface for different quantum chemistry software.
    All specific implementations (PySCF, MACE, UMA, etc.) should inherit from this class.
    
    Attributes:
        implemented_properties (list): Properties that can be calculated.
        config (dict): Configuration dictionary for the calculator.
        method: The underlying method/model object (software-specific).
    """
    
    implemented_properties = ["energy", "forces"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the base calculator.
        
        Args:
            config: Configuration dictionary containing calculator parameters.
            **kwargs: Additional keyword arguments for ASE Calculator.
        """
        Calculator.__init__(self, **kwargs)
        self.config = config or {}
        self.method = None
        self._hessian_cache = {}
        self._converged = True
        
    # ==================== Abstract Methods (Must be implemented) ====================
    
    @abstractmethod
    def _build_method(self, atoms: Atoms) -> Any:
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
    def _compute_energy_and_forces(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces for the given atoms.
        
        This is the core calculation method that must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Tuple of (energy in eV, forces in eV/Angstrom with shape (N, 3)).
        """
        pass
    
    @abstractmethod
    def _compute_hessian_impl(self, atoms: Atoms) -> np.ndarray:
        """
        Compute the Hessian matrix.
        
        This is software-specific and must be implemented by each subclass.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Hessian matrix with shape (3N, 3N) in eV/Angstrom^2.
        """
        pass
    
    # ==================== Common Methods (Implemented in base class) ====================
    
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: list = all_changes,
    ):
        """
        Perform the calculation.
        
        This method is called by ASE to compute properties.
        
        Args:
            atoms: ASE Atoms object (uses self.atoms if None).
            properties: List of properties to calculate.
            system_changes: List of changes since last calculation.
        """
        if properties is None:
            properties = self.implemented_properties
            
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Build method if not already built
        if self.method is None:
            self.method = self._build_method(atoms)
        
        # Compute energy and forces
        energy, forces = self._compute_energy_and_forces(atoms)
        
        # Store results
        self.results["energy"] = energy
        self.results["forces"] = forces
        
    def compute_hessian(
        self, 
        atoms: Atoms,
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
        atoms: Atoms,
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
        
    def is_converged(self) -> bool:
        """
        Check if the last calculation converged.
        
        Returns:
            True if converged, False otherwise.
        """
        return self._converged
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        self.config[key] = value
        
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
    
    def __repr__(self) -> str:
        """String representation of the calculator."""
        return f"{self.__class__.__name__}(config={self.config})"
