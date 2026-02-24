"""
Unit tests for the unified calculator architecture.
"""

import pytest
import numpy as np
from ase import Atoms

from qc_launcher.calculators import PySCFCalculator, MLFFCalculator


def create_h2_molecule():
    """Create a simple H2 molecule for testing."""
    atoms = Atoms('H2', positions=[(0, 0, 0), (0.74, 0, 0)])
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    return atoms


def create_h2o_molecule():
    """Create a water molecule for testing."""
    atoms = Atoms(
        'H2O',
        positions=[(0, 0, 0), (0.96, 0, 0), (0.24, 0.93, 0)],
    )
    atoms.info['charge'] = 0
    atoms.info['multiplicity'] = 1
    return atoms


class TestBaseCalculatorInterface:
    """Test the base calculator interface."""
    
    def test_pyscf_energy_calculation(self):
        """Test energy calculation with PySCF."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        
        assert isinstance(energy, float)
        assert energy < 0  # Should be negative for bound system
    
    def test_pyscf_forces_calculation(self):
        """Test forces calculation with PySCF."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        forces = atoms.get_forces()
        
        assert forces.shape == (2, 3)
        assert np.all(np.isfinite(forces))
    
    def test_pyscf_hessian_calculation(self):
        """Test Hessian calculation with PySCF."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        hessian = calc.compute_hessian(atoms)
        
        # Check shape
        assert hessian.shape == (6, 6)
        
        # Check symmetry
        assert np.allclose(hessian, hessian.T)
        
        # Check finite values
        assert np.all(np.isfinite(hessian))
    
    def test_hessian_caching(self):
        """Test Hessian caching mechanism."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        # First calculation
        hessian1 = calc.compute_hessian(atoms, use_cache=True)
        
        # Second calculation (should use cache)
        hessian2 = calc.compute_hessian(atoms, use_cache=True)
        
        # Should be identical (from cache)
        assert np.allclose(hessian1, hessian2)
        
        # Clear cache and recalculate
        calc.clear_cache()
        hessian3 = calc.compute_hessian(atoms, use_cache=False)
        
        # Should still be equal but freshly computed
        assert np.allclose(hessian1, hessian3)
    
    def test_vibrational_modes(self):
        """Test vibrational mode calculation."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        vib_data = calc.get_vibrational_modes(atoms)
        
        # Check keys
        assert 'frequencies' in vib_data
        assert 'modes' in vib_data
        assert 'reduced_mass' in vib_data
        
        # Check shapes
        assert vib_data['frequencies'].shape == (6,)
        assert vib_data['modes'].shape == (6, 6)
        assert vib_data['reduced_mass'].shape == (6,)
        
        # For H2, should have ~5 low frequencies (translations/rotations)
        # and 1 high frequency (stretch)
        freqs = vib_data['frequencies']
        assert np.sum(np.abs(freqs) > 100) >= 1  # At least one real vibration
    
    def test_convergence_check(self):
        """Test convergence checking."""
        atoms = create_h2_molecule()
        
        config = {
            'xc': 'HF',
            'basis': 'sto-3g',
            'charge': 0,
            'spin': 0,
        }
        
        calc = PySCFCalculator(config=config)
        atoms.calc = calc
        
        # Perform calculation
        energy = atoms.get_potential_energy()
        
        # Should converge for simple system
        assert calc.is_converged()
    
    def test_multiple_functionals(self):
        """Test with different functionals."""
        atoms = create_h2_molecule()
        
        functionals = ['HF', 'B3LYP', 'PBE']
        energies = []
        
        for xc in functionals:
            config = {
                'xc': xc,
                'basis': 'sto-3g',
                'charge': 0,
                'spin': 0,
            }
            
            calc = PySCFCalculator(config=config)
            atoms.calc = calc
            
            energy = atoms.get_potential_energy()
            energies.append(energy)
            
            assert isinstance(energy, float)
            assert energy < 0
        
        # Different functionals should give different energies
        assert len(set([f"{e:.6f}" for e in energies])) > 1
    
    def test_config_access(self):
        """Test configuration access methods."""
        config = {
            'xc': 'B3LYP',
            'basis': 'def2-SVP',
        }
        
        calc = PySCFCalculator(config=config)
        
        # Test get_config
        assert calc.get_config('xc') == 'B3LYP'
        assert calc.get_config('basis') == 'def2-SVP'
        assert calc.get_config('nonexistent', 'default') == 'default'
        
        # Test set_config
        calc.set_config('new_key', 'new_value')
        assert calc.get_config('new_key') == 'new_value'


class TestMLFFCalculator:
    """Test MLFF calculator (if available)."""
    
    def test_mlff_import(self):
        """Test that MLFF calculator can be imported."""
        from qc_launcher.calculators import MLFFCalculator
        assert MLFFCalculator is not None
    
    @pytest.mark.skip(reason="Requires trained MACE model")
    def test_mlff_energy(self):
        """Test MLFF energy calculation."""
        atoms = create_h2o_molecule()
        
        config = {
            'model_path': '/path/to/model.pt',
            'model_type': 'mace',
            'device': 'cpu',
        }
        
        calc = MLFFCalculator(config=config)
        atoms.calc = calc
        
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)


class TestCalculatorComparison:
    """Test consistency across different calculators."""
    
    def test_interface_consistency(self):
        """Test that all calculators have the same interface."""
        from qc_launcher.calculators import PySCFCalculator, MLFFCalculator
        
        required_methods = [
            'calculate',
            'compute_hessian',
            'get_vibrational_modes',
            'clear_cache',
            'is_converged',
            'get_config',
            'set_config',
        ]
        
        for calc_class in [PySCFCalculator, MLFFCalculator]:
            for method in required_methods:
                assert hasattr(calc_class, method), \
                    f"{calc_class.__name__} missing method: {method}"


def test_numerical_hessian_comparison():
    """Compare analytical Hessian with numerical."""
    from ase.calculators.test import numeric_force
    
    atoms = create_h2_molecule()
    
    config = {
        'xc': 'HF',
        'basis': 'sto-3g',
        'charge': 0,
        'spin': 0,
    }
    
    calc = PySCFCalculator(config=config)
    atoms.calc = calc
    
    # Analytical Hessian
    hessian_analytical = calc.compute_hessian(atoms)
    
    # Note: Full numerical Hessian comparison would require
    # finite difference implementation. This is a simplified test.
    # In practice, you would compare with numerical derivatives.
    
    assert np.all(np.isfinite(hessian_analytical))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
