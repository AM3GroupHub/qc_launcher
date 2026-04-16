from typing import Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.units import Hartree
try:
    from nvalchemiops.torch.neighbors import neighbor_list
    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False
if NVALCHEMI_AVAILABLE:
    def nv_get_neighborhood(
        positions: np.ndarray,
        cutoff: float,
        pbc: Optional[Tuple[bool, bool, bool]] = None,
        cell: Optional[np.ndarray] = None,
        true_self_interaction=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positions_tensor = torch.from_numpy(positions).float()  # [num_positions, 3]
        cell_tensor = torch.from_numpy(cell).float() if cell is not None else None
        pbc_tensor = torch.tensor(pbc, dtype=torch.bool) if pbc is not None else torch.zeros(3, dtype=torch.bool)
        if torch.any(pbc_tensor):
            edge_index_tensor, _, unit_shifts_tensor = neighbor_list(
                positions=positions_tensor,
                cutoff=cutoff,
                cell=cell_tensor,
                pbc=pbc_tensor,
                return_neighbor_list=True,
            )
            shifts_tensor = torch.dot(unit_shifts_tensor, cell_tensor)  # [n_edges, 3]
        else:
            edge_index_tensor, _ = neighbor_list(
                positions=positions_tensor,
                cutoff=cutoff,
                cell=None,
                pbc=None,
                return_neighbor_list=True,
            )
            unit_shifts_tensor = torch.zeros(edge_index_tensor.shape[1], 3, dtype=torch.int32)  # [n_edges, 3]
            shifts_tensor = torch.zeros_like(unit_shifts_tensor, dtype=torch.float)  # [n_edges, 3]
        # copy from original get_neighborhood
        pbc_x = pbc[0]
        pbc_y = pbc[1]
        pbc_z = pbc[2]
        identity = np.identity(3, dtype=float)
        max_positions = np.max(np.absolute(positions)) + 1
        # Extend cell in non-periodic directions
        # For models with more than 5 layers, the multiplicative constant needs to be increased.
        # temp_cell = np.copy(cell)
        if not pbc_x:
            cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
        if not pbc_y:
            cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
        if not pbc_z:
            cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

        return edge_index_tensor.numpy(), shifts_tensor.numpy(), unit_shifts_tensor.numpy(), cell
    import mace.data.atomic_data
    mace.data.atomic_data.get_neighborhood = nv_get_neighborhood

from mace.calculators import mace_omol, mace_mp
from mace.calculators import MACECalculator
from pyscf import gto

from .base_driver import BaseDriver


class MACEDriver(BaseDriver):
    def __init__(
        self,
        atoms: Atoms,
        config: dict,
        **kwargs,
    ):
        super().__init__(atoms=atoms, config=config, **kwargs)
        self.calc: MACECalculator = None
        self._add_spin_tag()

    def _add_spin_tag(self):
        # the tag of "multiplicity" in MACE is `spin`, which is different from the `spin` in pyscf
        self.atoms.info["spin"] = self.atoms.info.get("multiplicity", 1)
    
    def update_atoms(self, atoms: Optional[Atoms]) -> bool:
        updated = super().update_atoms(atoms)
        if updated:
            self._add_spin_tag()
        return updated

    def build_calc(self) -> MACECalculator:
        model_path = self.config["model_path"]
        device = self.config.get("device", None)
        precision = self.config.get("precision", "float64")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_type = self.config.get("model_type", "omol")
        head = self.config.get("head", None)  # only for mp models
        if model_type == "omol":
            return mace_omol(model=model_path, device=device, default_dtype=precision)
        elif model_type == "mp":
            dispersion = self.config.get("dispersion", False)
            dispersion_xc = self.config.get("dispersion_xc", "pbe")
            damping = self.config.get("damping", "bj")
            return mace_mp(
                model=model_path, device=device, precision=precision, head=head,
                dispersion=dispersion, dispersion_xc=dispersion_xc, damping=damping,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def to_ase_calc(self) -> MACECalculator:
        if self.calc is None:
            self.calc = self.build_calc()
        return self.calc

    def to_pyscf_mf(self):
        mol = gto.M(
            atom=[(symb, coord) for symb, coord in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())],
            charge=self.atoms.info.get("charge", 0),
            spin=self.atoms.info.get("multiplicity", 1) - 1,
        )
        if self.calc is None:
            self.calc = self.build_calc()
        e_tot = self.calc.get_potential_energy(self.atoms) / Hartree  # convert from eV to Hartree
        dummy_mf = SimpleNamespace(mol=mol, e_tot=e_tot)
        return dummy_mf

    def _compute_energy_impl(self, atoms: Optional[Atoms] = None) -> Tuple[float, str]:
        if self.calc is None:
            self.calc = self.build_calc()
        self.update_atoms(atoms)
        energy = self.calc.get_potential_energy(self.atoms)
        return energy, "eV"

    def _compute_forces_impl(self, atoms):
        if self.calc is None:
            self.calc = self.build_calc()
        self.update_atoms(atoms)
        forces = self.calc.get_forces(self.atoms)
        return forces, "eV/Ang"

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
    ) -> np.ndarray:
        if self.calc is None:
            self.calc = self.build_calc()
        self.update_atoms(atoms)
        natm = len(self.atoms)
        hessian = self.calc.get_hessian(self.atoms).reshape(natm * 3, natm * 3)
        return hessian
    