"""
Machine Learning Force Field (MLFF) calculator implementation.
Supports MACE, UMA, and other PyTorch-based models.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
from ase import Atoms, units

from qc_launcher.common.base_calculator import BaseQCCalculator


class MLFFCalculator(BaseQCCalculator):
    """
    Machine Learning Force Field calculator.
    
    This calculator supports PyTorch-based ML models like MACE, UMA, etc.
    Uses autograd to compute Hessian from forces.
    
    Example:
        >>> import mace
        >>> model = mace.calculators.MACECalculator(model_path='model.pt')
        >>> config = {'device': 'cuda', 'dtype': 'float64'}
        >>> calc = MLFFCalculator(model=model, config=config)
        >>> atoms.calc = calc
    """
    
    implemented_properties = ["energy", "forces"]
    
    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        dtype: str = "float64",
        **kwargs
    ):
        """
        Initialize MLFF calculator.
        
        Args:
            model: PyTorch model or ASE calculator.
            config: Configuration dictionary.
            device: Device for computation ('cpu' or 'cuda').
            dtype: Data type ('float32' or 'float64').
            **kwargs: Additional arguments for base calculator.
        """
        super().__init__(config=config, **kwargs)
        
        self.model = model
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        
    def _build_method(self, atoms: Atoms) -> Any:
        """
        Build/load the ML model.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            The ML model.
        """
        if self.model is not None:
            return self.model
            
        # Load model from config
        model_path = self.get_config("model_path")
        if model_path is None:
            raise ValueError("Either 'model' or 'model_path' must be provided")
            
        # Try to load MACE model (can be extended for other models)
        model_type = self.get_config("model_type", "mace")
        
        if model_type.lower() == "mace":
            from mace.calculators import MACECalculator
            model = MACECalculator(
                model_paths=model_path,
                device=str(self.device),
                default_dtype=str(self.dtype).split('.')[-1],
            )
        elif model_type.lower() == "uma":
            # Add UMA model loading here
            raise NotImplementedError("UMA model loading not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return model
    
    def _compute_energy_and_forces(self, atoms: Atoms) -> Tuple[float, np.ndarray]:
        """
        Compute energy and forces using ML model.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Tuple of (energy in eV, forces in eV/Angstrom).
        """
        # If model is an ASE calculator, use it directly
        if hasattr(self.model, 'get_potential_energy'):
            atoms_copy = atoms.copy()
            atoms_copy.calc = self.model
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()
            return energy, forces
        
        # Otherwise, use PyTorch model directly
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )
        
        # Forward pass (model-specific, may need adaptation)
        output = self.model(positions, atoms.get_atomic_numbers())
        energy = output['energy'].item()
        forces = output['forces'].detach().cpu().numpy()
        
        return energy, forces
    
    def _compute_hessian_impl(self, atoms: Atoms) -> np.ndarray:
        """
        Compute Hessian using autograd.
        
        Args:
            atoms: ASE Atoms object.
            
        Returns:
            Hessian matrix in eV/Angstrom^2 with shape (3N, 3N).
        """
        positions = torch.tensor(
            atoms.get_positions(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )
        
        # Get forces with grad enabled
        if hasattr(self.model, 'get_forces'):
            # For ASE calculator interface
            atoms_copy = atoms.copy()
            atoms_copy.positions = positions.detach().cpu().numpy()
            atoms_copy.calc = self.model
            
            # Need to recompute with torch
            positions.requires_grad = True
            energy = self._get_torch_energy(atoms_copy, positions)
            forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
        else:
            # Direct model interface
            output = self.model(positions, atoms.get_atomic_numbers())
            forces = output['forces']
        
        # Compute Hessian using autograd
        hessian = self._compute_hessian_from_forces(positions, forces)
        
        return hessian.detach().cpu().numpy()
    
    def _compute_hessian_from_forces(
        self,
        positions: torch.Tensor,
        forces: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hessian from forces using autograd.
        
        Args:
            positions: Positions tensor with shape (N, 3).
            forces: Forces tensor with shape (N, 3).
            
        Returns:
            Hessian tensor with shape (3N, 3N).
        """
        forces_flat = forces.view(-1)
        n_hess_elements = forces_flat.shape[0]  # 3N
        
        def get_vjp(v):
            return torch.autograd.grad(
                outputs=-forces_flat,
                inputs=positions,
                grad_outputs=v,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
        
        I_N = torch.eye(n_hess_elements, device=positions.device, dtype=positions.dtype)
        
        try:
            # Use vmap for efficiency
            chunk_size = 1 if n_hess_elements < 64 else 16
            hessian = torch.vmap(get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size)(I_N)[0]
        except RuntimeError:
            # Fallback to loop if vmap fails
            hessian = []
            for grad_elem in forces_flat:
                hess_row = torch.autograd.grad(
                    outputs=-grad_elem,
                    inputs=positions,
                    grad_outputs=torch.ones_like(grad_elem),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                hess_row = hess_row.detach()
                hessian.append(hess_row)
            hessian = torch.stack(hessian)
        
        hessian = hessian.view(n_hess_elements, n_hess_elements)
        return hessian
    
    def _get_torch_energy(self, atoms: Atoms, positions: torch.Tensor) -> torch.Tensor:
        """
        Helper to get energy as torch tensor for autograd.
        
        Args:
            atoms: ASE Atoms object.
            positions: Positions tensor.
            
        Returns:
            Energy as torch tensor.
        """
        # This is a placeholder - actual implementation depends on model interface
        raise NotImplementedError("Direct torch energy computation not implemented")
