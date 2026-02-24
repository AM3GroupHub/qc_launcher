"""
Usage examples for the unified calculator interface.

This file demonstrates how to use the base calculator class
with different quantum chemistry software.
"""

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

# Example 1: Using PySCF Calculator
def example_pyscf():
    """Example using PySCF calculator."""
    from qc_launcher.calculators import PySCFCalculator
    
    # Create a simple water molecule
    atoms = Atoms(
        'H2O',
        positions=[(0, 0, 0), (0.96, 0, 0), (0.24, 0.93, 0)],
    )
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    
    # Configure calculator
    config = {
        'xc': 'B3LYP',
        'basis': 'def2-SVP',
        'charge': 0,
        'spin': 0,
        'inputfile': 'h2o.xyz',  # Just for reference
    }
    
    # Create calculator
    calc = PySCFCalculator(config=config, soscf=True)
    atoms.calc = calc
    
    # Compute properties
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    
    print(f"Energy: {energy:.6f} eV")
    print(f"Forces:\n{forces}")
    
    # Compute Hessian
    hessian = calc.compute_hessian(atoms)
    print(f"Hessian shape: {hessian.shape}")
    
    # Get vibrational modes
    vib_data = calc.get_vibrational_modes(atoms)
    print(f"Frequencies (cm^-1): {vib_data['frequencies'][6:]}")  # Skip translations/rotations
    
    return atoms, calc


# Example 2: Using MLFF Calculator (MACE)
def example_mace():
    """Example using MACE calculator."""
    from qc_launcher.calculators import MLFFCalculator
    
    # Create atoms
    atoms = Atoms(
        'H2O',
        positions=[(0, 0, 0), (0.96, 0, 0), (0.24, 0.93, 0)],
    )
    
    # Configure calculator
    config = {
        'model_path': '/path/to/mace_model.pt',
        'model_type': 'mace',
        'device': 'cpu',
        'dtype': 'float64',
    }
    
    try:
        # Create calculator
        calc = MLFFCalculator(config=config)
        atoms.calc = calc
        
        # Compute properties (same interface as PySCF!)
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        print(f"Energy: {energy:.6f} eV")
        print(f"Forces:\n{forces}")
        
        # Compute Hessian (same interface!)
        hessian = calc.compute_hessian(atoms)
        print(f"Hessian shape: {hessian.shape}")
        
        return atoms, calc
    except Exception as e:
        print(f"MACE example failed (expected if model not available): {e}")
        return None, None


# Example 3: Optimization with unified interface
def example_optimization(calc_type='pyscf'):
    """
    Example showing geometry optimization works with any calculator.
    
    Args:
        calc_type: Type of calculator ('pyscf' or 'mace').
    """
    from qc_launcher.calculators import PySCFCalculator, MLFFCalculator
    
    # Create atoms
    atoms = Atoms(
        'H2O',
        positions=[(0, 0, 0), (1.0, 0, 0), (0.5, 1.0, 0)],  # Distorted geometry
    )
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    
    # Choose calculator based on type
    if calc_type == 'pyscf':
        config = {
            'xc': 'B3LYP',
            'basis': 'def2-SVP',
            'charge': 0,
            'spin': 0,
        }
        calc = PySCFCalculator(config=config)
    elif calc_type == 'mace':
        config = {
            'model_path': '/path/to/model.pt',
            'model_type': 'mace',
        }
        calc = MLFFCalculator(config=config)
    else:
        raise ValueError(f"Unknown calculator type: {calc_type}")
    
    atoms.calc = calc
    
    # Optimize geometry (same code works for any calculator!)
    opt = BFGS(atoms)
    opt.run(fmax=0.01)
    
    print(f"Optimized with {calc_type}")
    print(f"Final positions:\n{atoms.get_positions()}")
    
    return atoms, calc


# Example 4: Hessian computation with different methods
def example_hessian_comparison():
    """Compare Hessian computation from different calculators."""
    from qc_launcher.calculators import PySCFCalculator
    
    atoms = Atoms(
        'H2O',
        positions=[(0, 0, 0), (0.96, 0, 0), (0.24, 0.93, 0)],
    )
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    
    # PySCF with different functionals
    for xc in ['HF', 'B3LYP', 'PBE']:
        config = {
            'xc': xc,
            'basis': 'def2-SVP',
            'charge': 0,
            'spin': 0,
        }
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        # Compute Hessian
        hessian = calc.compute_hessian(atoms)
        
        # Get frequencies
        vib_data = calc.get_vibrational_modes(atoms, hessian=hessian)
        freqs = vib_data['frequencies'][6:]  # Skip first 6 (translations/rotations)
        
        print(f"{xc:10s}: Frequencies = {freqs}")


# Example 5: Using the calculator factory pattern
def create_calculator(software: str, config: dict):
    """
    Factory function to create calculator based on software type.
    
    Args:
        software: Software name ('pyscf', 'mace', 'uma', etc.).
        config: Configuration dictionary.
        
    Returns:
        Calculator instance.
    """
    from qc_launcher.calculators import PySCFCalculator, MLFFCalculator
    
    software = software.lower()
    
    if software == 'pyscf':
        return PySCFCalculator(config=config)
    elif software in ['mace', 'uma', 'mlff']:
        return MLFFCalculator(config=config)
    else:
        raise ValueError(f"Unknown software: {software}")


def example_factory():
    """Example using factory pattern."""
    atoms = Atoms('H2', positions=[(0, 0, 0), (0.74, 0, 0)])
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    
    # Configuration can be read from YAML file
    config = {
        'xc': 'B3LYP',
        'basis': 'def2-SVP',
        'charge': 0,
        'spin': 0,
    }
    
    # Create calculator using factory
    calc = create_calculator('pyscf', config)
    atoms.calc = calc
    
    energy = atoms.get_potential_energy()
    print(f"H2 energy: {energy:.6f} eV")
    
    return atoms, calc


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Basic PySCF calculation")
    print("=" * 60)
    try:
        example_pyscf()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Example 2: MACE calculation")
    print("=" * 60)
    example_mace()
    
    print("\n" + "=" * 60)
    print("Example 5: Factory pattern")
    print("=" * 60)
    try:
        example_factory()
    except Exception as e:
        print(f"Error: {e}")
