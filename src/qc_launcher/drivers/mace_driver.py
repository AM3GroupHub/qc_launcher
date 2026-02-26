import os
from typing import Optional, Literal

import numpy as np
import torch
from ase import Atoms
from mace.calculators import mace_omol, mace_mp
from mace.calculators import MACECalculator

from .base_driver import BaseDriver


class MACEDriver(BaseDriver):
    def __init__(
        self,
        atoms: Atoms,
        config: dict,
    ):
        super().__init__(atoms=atoms, config=config)
        self.calc: MACECalculator = None
        
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

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
        hess_format: Literal["pyscf", "ase"] = "ase",
    ) -> np.ndarray:
        if self.calc is None:
            self.calc = self.build_calc()
        self.update_atoms(atoms)
        natm = len(self.atoms)
        hessian = self.calc.get_hessian(self.atoms).reshape(natm * 3, natm * 3)
        hessian = self._convert_hessian_format(hessian=hessian, hess_format=hess_format)
        return hessian
    