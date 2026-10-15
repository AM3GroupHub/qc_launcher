"""FAIRChem UMA driver (fairchem-core)."""

from __future__ import annotations

import os

import numpy as np
import torch
from ase import Atoms
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets import data_list_collater
from omegaconf import OmegaConf

from .calc_driver import CalculatorDriver


class UMADriver(CalculatorDriver):
    """Config keys (defaults): ``model_path`` (required), ``atom_refs_path`` (``iso_atom_elem_refs.yaml`` next
    to the model, if present), ``task_name`` ('omol'), ``device`` (auto), ``finite_diff_eps`` [Angstrom]
    (5e-3), ``batch_size`` (128) for the finite-difference Hessian.
    """

    def build_calc(self) -> FAIRChemCalculator:
        model_path = self.config["model_path"]
        atom_refs_path = self.config.get(
            "atom_refs_path", os.path.join(os.path.dirname(model_path), "iso_atom_elem_refs.yaml")
        )
        atom_refs = OmegaConf.load(atom_refs_path) if os.path.exists(atom_refs_path) else None
        predictor = pretrained_mlip.load_predict_unit(
            model_path, device=self.resolve_device(self.config), atom_refs=atom_refs
        )
        return FAIRChemCalculator(predictor, task_name=self.config.get("task_name", "omol"))

    def _compute_hessian_impl(self, atoms: Atoms | None) -> np.ndarray:
        """Central differences of the predicted forces, batched through the predictor; (3N, 3N) in eV/Ang^2."""
        self.update_atoms(atoms)
        eps = float(self.config.get("finite_diff_eps", 5e-3))
        batch_size = int(self.config.get("batch_size", 128))
        natm = len(self.atoms)
        data_list = []
        for idx in range(3 * natm):
            i, j = divmod(idx, 3)
            for sign in (1.0, -1.0):
                displaced = self.atoms.copy()
                displaced.positions[i, j] += sign * eps
                data_list.append(self.calc.a2g(displaced))
        forces = []
        for start in range(0, len(data_list), batch_size):
            batch = data_list_collater(data_list[start : start + batch_size], otf_graph=True)
            forces.append(self.calc.predictor.predict(batch)["forces"].detach().cpu())
        forces = torch.cat(forces).reshape(3 * natm, 2, 3 * natm).numpy()  # [displaced coord, +/-, force component]
        hessian = (forces[:, 1, :] - forces[:, 0, :]) / (2.0 * eps)  # d(-F_col)/d(x_row)
        return 0.5 * (hessian + hessian.T)
