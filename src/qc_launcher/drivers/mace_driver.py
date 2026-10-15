"""MACE driver (mace-torch), with an optional nvalchemiops neighbour list."""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator, mace_mp, mace_omol

from .calc_driver import CalculatorDriver

NEIGHBOR_LISTS = ("mace", "nvalchemi")


def nvalchemi_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: np.ndarray | None = None,
    true_self_interaction: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop-in replacement for ``mace.data.neighborhood.get_neighborhood`` backed by nvalchemiops.

    Returns ``(edge_index, shifts, unit_shifts, cell)`` with the same conventions as MACE: shifts are
    ``unit_shifts @ cell`` and the cell is extended along non-periodic directions.
    """
    from nvalchemiops.torch.neighbors import neighbor_list

    pbc = (False, False, False) if pbc is None else tuple(bool(p) for p in pbc)
    cell = np.identity(3) if cell is None or not np.any(cell) else np.array(cell, dtype=float)
    positions_t = torch.as_tensor(np.asarray(positions), dtype=torch.float64)
    if any(pbc):
        cell_t = torch.as_tensor(cell, dtype=positions_t.dtype)
        edge_index, _, unit_shifts = neighbor_list(
            positions=positions_t, cutoff=cutoff, cell=cell_t, pbc=torch.tensor(pbc), return_neighbor_list=True
        )
        shifts = unit_shifts.to(positions_t.dtype) @ cell_t
    else:
        edge_index, _ = neighbor_list(positions=positions_t, cutoff=cutoff, return_neighbor_list=True)
        unit_shifts = torch.zeros(edge_index.shape[1], 3, dtype=torch.int32)
        shifts = torch.zeros(edge_index.shape[1], 3, dtype=positions_t.dtype)
    # extend the cell in non-periodic directions, as MACE does
    max_positions = np.max(np.abs(positions)) + 1
    for axis in range(3):
        if not pbc[axis]:
            cell[axis] = 0.0
            cell[axis, axis] = max_positions * 5 * cutoff
    return edge_index.numpy(), shifts.numpy(), unit_shifts.numpy(), cell


def set_neighbor_list(kind: str) -> None:
    """Select the neighbour-list implementation MACE uses process-wide ('mace' or 'nvalchemi')."""
    import mace.data.atomic_data as atomic_data
    import mace.data.neighborhood as neighborhood

    if kind == "nvalchemi":
        atomic_data.get_neighborhood = nvalchemi_neighborhood
    elif kind == "mace":
        atomic_data.get_neighborhood = neighborhood.get_neighborhood
    else:
        raise ValueError(f"Unsupported neighbor_list '{kind}'. Use one of {NEIGHBOR_LISTS}.")


class MACEDriver(CalculatorDriver):
    """Config keys (defaults): ``model_path`` (required), ``model_type`` ('omol' | 'mp'), ``device`` (auto),
    ``precision`` ('float64'), ``head`` / ``dispersion`` / ``dispersion_xc`` / ``damping`` (mp models only),
    ``neighbor_list`` ('nvalchemi' when nvalchemiops is installed, else 'mace').
    """

    def build_calc(self) -> MACECalculator:
        default_nlist = "nvalchemi" if importlib.util.find_spec("nvalchemiops") is not None else "mace"
        set_neighbor_list(self.config.get("neighbor_list", default_nlist))
        model_path = self.config["model_path"]
        device = self.resolve_device(self.config)
        precision = self.config.get("precision", "float64")
        model_type = self.config.get("model_type", "omol")
        if model_type == "omol":
            return mace_omol(model=model_path, device=device, default_dtype=precision)
        if model_type == "mp":
            return mace_mp(
                model=model_path,
                device=device,
                default_dtype=precision,
                head=self.config.get("head"),
                dispersion=self.config.get("dispersion", False),
                dispersion_xc=self.config.get("dispersion_xc", "pbe"),
                damping=self.config.get("damping", "bj"),
            )
        raise ValueError(f"Unsupported model_type '{model_type}'. Use 'omol' or 'mp'.")

    def _compute_hessian_impl(self, atoms: Atoms | None) -> np.ndarray:
        self.update_atoms(atoms)
        natm = len(self.atoms)
        return np.asarray(self.calc.get_hessian(self.atoms)).reshape(3 * natm, 3 * natm)
