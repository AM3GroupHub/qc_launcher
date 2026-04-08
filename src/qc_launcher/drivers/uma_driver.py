import os
from typing import Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.units import Hartree
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets import data_list_collater
from omegaconf import OmegaConf
from pyscf import gto

from .base_driver import BaseDriver


class UMADriver(BaseDriver):
    def __init__(
        self,
        atoms: Atoms,
        config: dict,
        **kwargs,
    ):
        super().__init__(atoms=atoms, config=config, **kwargs)
        self.calc: FAIRChemCalculator = None
        self._add_spin_tag()

    def _add_spin_tag(self):
        # the tag of "multiplicity" in UMA is `spin`, which is different from the `spin` in pyscf
        self.atoms.info["spin"] = self.atoms.info.get("multiplicity", 1)

    def update_atoms(self, atoms: Optional[Atoms]) -> bool:
        updated = super().update_atoms(atoms)
        if updated:
            self._add_spin_tag()
        return updated

    def build_calc(self) -> FAIRChemCalculator:
        model_path = self.config["model_path"]
        atom_refs_path = self.config.get(
            "atom_refs_path", os.path.join(os.path.dirname(model_path), "iso_atom_elem_refs.yaml")
        )
        atom_refs = OmegaConf.load(atom_refs_path) if os.path.exists(atom_refs_path) else None
        device = self.config.get("device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.load_predict_unit(model_path, device=device, atom_refs=atom_refs)
        task_name = self.config.get("task_name", "omol")
        return FAIRChemCalculator(predictor, task_name=task_name)

    def to_ase_calc(self) -> FAIRChemCalculator:
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
    
    def _compute_forces_impl(self, atoms: Optional[Atoms] = None) -> Tuple[np.ndarray, str]:
        if self.calc is None:
            self.calc = self.build_calc()
        self.update_atoms(atoms)
        forces = self.calc.get_forces(self.atoms)
        return forces, "eV/Ang"

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
    ) -> np.ndarray:
        self.update_atoms(atoms)
        eps = self.config.get("finite_diff_eps", 5e-3)
        data_list = []
        if self.calc is None:
            self.calc = self.build_calc()
        for i in range(len(self.atoms)):
            for j in range(3):
                displaced_plus = self.atoms.copy()
                displaced_minus = self.atoms.copy()
                displaced_plus.positions[i, j] += eps
                displaced_minus.positions[i, j] -= eps
                data_plus = self.calc.a2g(displaced_plus)
                data_minus = self.calc.a2g(displaced_minus)
                data_list.extend([data_plus, data_minus])
        # batch and predict
        forces_list = []
        batch_size = self.config.get("batch_size", 128)
        for i in range(0, len(data_list), batch_size):
            if i + batch_size > len(data_list):
                data_list_batch = data_list[i:]
            else:
                data_list_batch = data_list[i:i+batch_size]
            batch = data_list_collater(data_list_batch, otf_graph=True)
            pred = self.calc.predictor.predict(batch)
            batch_forces = pred["forces"].detach()
            forces_list.append(batch_forces)
        forces = torch.cat(forces_list, dim=0).reshape(-1, len(self.atoms), 3)
        # calculated hessian using finite differences
        hessian = np.zeros((len(self.atoms) * 3, len(self.atoms) * 3))
        for i in range(len(self.atoms)):
            for j in range(3):
                idx = i * 3 + j
                forces_plus = forces[2 * idx].flatten().cpu().numpy()
                forces_minus = forces[2 * idx + 1].flatten().cpu().numpy()
                hessian[:, idx] = (forces_minus - forces_plus) / (2 * eps) # forces is the negative graidents
        return hessian
