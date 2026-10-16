"""tblite (xTB) driver: GFN1/GFN2-xTB energies, gradients and finite-difference Hessians."""

from __future__ import annotations

import ase.calculators.calculator
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree, kB
from tblite.ase import TBLite, _create_api_calculator
from tblite.exceptions import TBLiteRuntimeError
from tblite.interface import Calculator, Result

from qc_launcher.utils.config import rename_deprecated_keys

from .base_driver import BaseDriver, dummy_pyscf_mf

DEFAULT_EPSILON = 78.3553
ANNEALING_DEFAULTS = {"max_temp": 5000.0, "heating_step": 300.0, "annealing_step": 50.0}
GUESS_CODES = {"sad": 0, "eeq": 1}


def heating_annealing(
    xtb: Calculator,
    temp: float,
    max_temp: float = 5000.0,
    heating_step: float = 300.0,
    annealing_step: float = 50.0,
) -> Result | None:
    """Resolve xTB SCC convergence failures with an electronic-temperature annealing strategy.

    For systems with small HOMO-LUMO gaps the SCC iterations often oscillate. Raising the electronic
    temperature (Fermi smearing) broadens the occupations and aids convergence. The target state is
    recovered by first finding a converged high-temperature state and then cooling back down using the
    previous wavefunction as the restart.

    1. Heating: scan upwards from ``temp`` in steps of ``heating_step`` until the first converged state is
       found at ``temp_high``. Give up when ``max_temp`` is exceeded.
    2. Annealing: try ``temp`` directly; on failure bisect between the failed temperature and the last
       converged anchor. Stop successfully when ``temp`` converges, or unsuccessfully when the bisection gap
       drops below ``annealing_step``.

    Args:
        xtb: tblite API calculator supporting ``singlepoint()`` and ``set()``.
        temp: target electronic temperature in Kelvin (the calculator is reset to it on exit).
        max_temp, heating_step, annealing_step: in Kelvin.

    Returns:
        The converged :class:`Result` at ``temp``, or None if the target temperature was not reached.
    """
    kB_Hartree = kB / Hartree
    try:
        temp_high = None
        conv_res = None
        for t in np.arange(temp + heating_step, max_temp + heating_step, heating_step):
            t = float(t)
            xtb.set("temperature", t * kB_Hartree)
            try:
                conv_res = xtb.singlepoint()
                print(f"Heating: converged at {t:.2f} K.")
                temp_high = t
                break
            except TBLiteRuntimeError:
                print(f"Heating: failed to converge at {t:.2f} K.")
        if conv_res is None:
            print(f"Failed to converge even at {max_temp:.2f} K.")
            return None
        if abs(temp_high - temp) < 1e-3:
            return conv_res

        temp_try = temp
        while True:
            if abs(temp_try - temp_high) < annealing_step and temp_try > temp:
                print(f"Annealing limit reached. Best stable temperature: {temp_high:.2f} K.")
                return None
            xtb.set("temperature", temp_try * kB_Hartree)
            try:
                conv_res = xtb.singlepoint(conv_res)
                print(f"Annealing: converged at {temp_try:.2f} K.")
                temp_high = temp_try
                if abs(temp_try - temp) < 1e-3:
                    return conv_res
                next_temp = (temp_try + temp) / 2
                if abs(next_temp - temp_try) < annealing_step:
                    next_temp = temp
                temp_try = next_temp
            except TBLiteRuntimeError:
                print(f"Annealing: failed to converge at {temp_try:.2f} K.")
                next_temp = (temp_try + temp_high) / 2
                if abs(next_temp - temp_high) < annealing_step:
                    print(f"Annealing limit reached. Best stable temperature: {temp_high:.2f} K.")
                    return None
                temp_try = next_temp
    finally:
        xtb.set("temperature", temp * kB_Hartree)


def solvation_args(solvation: dict | None) -> tuple | None:
    """Translate the ``solvation`` block into tblite's ``(model, *args)`` tuple."""
    if solvation is None:
        return None
    solvation = rename_deprecated_keys(solvation, "driver.solvation")
    if "model" not in solvation:
        raise ValueError("The solvation block needs a 'model' (alpb, gbsa, cpcm, gbe or gb).")
    model = str(solvation["model"]).lower()
    if model in ("alpb", "gbsa"):
        return (model, solvation.get("solvent", "water"), solvation.get("solution_state", "gsolv"))
    if model == "cpcm":
        return (model, float(solvation.get("epsilon", DEFAULT_EPSILON)))
    if model in ("gbe", "gb"):
        return (model, float(solvation.get("epsilon", DEFAULT_EPSILON)), solvation.get("born_kernel", "still"))
    raise ValueError(f"Unsupported solvation method '{model}'. Use alpb, gbsa, cpcm, gbe or gb.")


def scc_settings(config: dict) -> dict:
    """SCC settings shared by the ASE calculator and the API calculator (temperatures in Kelvin)."""
    guess = str(config.get("guess", "sad")).lower()
    if guess not in GUESS_CODES:
        raise ValueError(f"Unsupported guess '{guess}'. Use 'sad' or 'eeq'.")
    electric_field = config.get("electric_field")
    return {
        "method": config.get("method", "GFN2-xTB"),
        "accuracy": float(config.get("accuracy", 1.0)),
        "guess": guess,
        "etemp": float(config.get("electronic_temperature", 298.15)),
        "max_iter": int(config.get("scf_max_iter", 250)),
        "mixer_damping": float(config.get("mixer_damping", 0.4)),
        "verbosity": int(config.get("verbosity", 0)),
        "electric_field": None if electric_field is None else np.asarray(electric_field, dtype=float),
        "spin_polarization": config.get("spin_polarization"),
        "solvation": solvation_args(config.get("solvation")),
    }


class TBLiteCalculator(TBLite):
    """tblite's ASE calculator plus the heating-annealing retry on SCC failure."""

    def __init__(
        self,
        atoms: Atoms | None = None,
        retry_annealing: bool = False,
        annealing_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(atoms, **kwargs)
        self.retry_annealing = retry_annealing
        self.annealing_config = {**ANNEALING_DEFAULTS, **(annealing_config or {})}

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = ase.calculators.calculator.all_changes,
    ) -> None:
        if not properties:
            properties = ["energy"]
        ase.calculators.calculator.Calculator.calculate(self, atoms, properties, system_changes)
        self._check_api_calculator(system_changes)
        if self._xtb is None:
            self._xtb = _create_api_calculator(self.atoms, self.parameters)

        try:
            self._res = self._xtb.singlepoint(self._res)
        except RuntimeError as exc:
            print(exc)
            self._res = None
            if self.retry_annealing:
                # parameters.electronic_temperature is in Kelvin, as heating_annealing expects
                self._res = heating_annealing(
                    self._xtb, temp=self.parameters.electronic_temperature, **self.annealing_config
                )
        if self._res is None:
            raise RuntimeError("xTB calculation failed to converge.")

        self.results["energy"] = self._res["energy"] * Hartree
        self.results["energies"] = self._res["energies"] * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self._res["gradient"] * Hartree / Bohr
        self.results["charges"] = self._res["charges"]
        self.results["dipole"] = self._res["dipole"] * Bohr
        self.results["bond-orders"] = self._res["bond-orders"]
        if self.atoms.pbc.any():  # the stress tensor only exists for periodic systems
            stress = self._res["virial"] * Hartree / self.atoms.get_volume()
            self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]


class TBLiteDriver(BaseDriver):
    """Config keys (defaults): ``method`` (GFN2-xTB), ``accuracy`` (1.0), ``guess`` (sad | eeq),
    ``electronic_temperature`` [K] (298.15), ``scf_max_iter`` (250), ``mixer_damping`` (0.4), ``verbosity`` (0),
    ``electric_field`` [V/Angstrom], ``spin_polarization``, ``solvation`` ({model: alpb | gbsa, solvent,
    solution_state} | {model: cpcm, epsilon} | {model: gbe | gb, epsilon, born_kernel}), ``retry_annealing``
    (False), ``annealing`` ({max_temp, heating_step, annealing_step} in K), ``finite_diff_step`` [Angstrom] (5e-3).

    SCC failures raise ``RuntimeError`` (after the annealing retry when enabled).
    """

    def __init__(self, atoms: Atoms | None, config: dict, **kwargs):
        super().__init__(atoms=atoms, config=config, **kwargs)
        self.settings = scc_settings(self.config)
        self.retry_annealing = bool(self.config.get("retry_annealing", False))
        self.annealing = {**ANNEALING_DEFAULTS, **(self.config.get("annealing") or {})}
        self.calc: TBLiteCalculator | None = None
        self.xtb: Calculator | None = None
        self._res_cache: Result | None = None

    def build_calc(self, atoms: Atoms | None = None) -> TBLiteCalculator:
        self.update_atoms(atoms)
        s = self.settings
        return TBLiteCalculator(
            method=s["method"],
            charge=self.atoms.info.get("charge", 0),
            multiplicity=self.atoms.info.get("multiplicity", 1),
            accuracy=s["accuracy"],
            initial_guess=s["guess"],
            electronic_temperature=s["etemp"],
            max_iterations=s["max_iter"],
            mixer_damping=s["mixer_damping"],
            electric_field=s["electric_field"],
            spin_polarization=s["spin_polarization"],
            solvation=s["solvation"],
            verbosity=s["verbosity"],
            retry_annealing=self.retry_annealing,
            annealing_config=self.annealing,
        )

    def build_xtb(self, atoms: Atoms | None = None) -> Calculator:
        self.update_atoms(atoms)
        s = self.settings
        periodic = self.atoms.get_pbc()
        xtb = Calculator(
            method=s["method"],
            numbers=self.atoms.get_atomic_numbers(),
            positions=self.atoms.get_positions() / Bohr,
            charge=self.atoms.info.get("charge", 0),
            uhf=self.atoms.info.get("multiplicity", 1) - 1,
            lattice=self.atoms.get_cell().array / Bohr if periodic.any() else None,
            periodic=periodic,
        )
        xtb.set("accuracy", s["accuracy"])
        xtb.set("guess", GUESS_CODES[s["guess"]])
        xtb.set("temperature", s["etemp"] * kB / Hartree)
        xtb.set("max-iter", s["max_iter"])
        xtb.set("mixer-damping", s["mixer_damping"])
        xtb.set("verbosity", s["verbosity"])
        if s["electric_field"] is not None:
            xtb.add("electric-field", s["electric_field"] * Bohr / Hartree)
        if s["spin_polarization"] is not None:
            xtb.add("spin-polarization", s["spin_polarization"])
        if s["solvation"] is not None:
            model, *args = s["solvation"]
            xtb.add(f"{model}-solvation", *args)
        return xtb

    def to_ase_calc(self) -> TBLiteCalculator:
        if self.calc is None:
            self.calc = self.build_calc()
        return self.calc

    def to_pyscf_mf(self):
        if self.atoms is None:
            raise ValueError("Atoms must be provided to build the pyscf stand-in.")
        return dummy_pyscf_mf(self.atoms, self.run_kernel()["energy"])

    def run_kernel(self, atoms: Atoms | None = None, use_cache: bool = True) -> Result:
        """Single point at the current positions; the previous result is reused as the SCC restart."""
        self.update_atoms(atoms)
        if use_cache and self._res_cache is not None and not self._positions_changed():
            return self._res_cache
        if self.xtb is None:
            self.xtb = self.build_xtb()
        else:
            periodic = self.atoms.get_pbc()
            self.xtb.update(
                positions=self.atoms.get_positions() / Bohr,
                lattice=self.atoms.get_cell().array / Bohr if periodic.any() else None,
            )
        try:
            res = self.xtb.singlepoint(self._res_cache if use_cache else None)
        except TBLiteRuntimeError as exc:
            if not self.retry_annealing:
                raise RuntimeError(f"xTB SCC did not converge: {exc}") from exc
            self.log(f"xTB SCC did not converge ({exc}); starting the heating-annealing procedure...")
            res = heating_annealing(self.xtb, temp=self.settings["etemp"], **self.annealing)
            if res is None:
                raise RuntimeError("xTB SCC did not converge even after heating-annealing.") from exc
        if use_cache:
            self._res_cache = res
            self._mark_positions()
        return res

    def _compute_energy_impl(self, atoms: Atoms | None = None) -> tuple[float, str]:
        return float(self.run_kernel(atoms)["energy"]), "Eh"

    def _compute_forces_impl(self, atoms: Atoms | None = None) -> tuple[np.ndarray, str]:
        return -np.asarray(self.run_kernel(atoms)["gradient"]), "Eh/Bohr"

    def _compute_hessian_impl(self, atoms: Atoms | None) -> np.ndarray:
        """Central differences of the gradient; (N, N, 3, 3) in Eh/Bohr^2, symmetrised."""
        self.update_atoms(atoms)
        natm = len(self.atoms)
        eps = float(self.config.get("finite_diff_step", 5e-3))  # Angstrom
        positions0 = self.atoms.get_positions().copy()
        hessian = np.zeros((natm, natm, 3, 3))
        try:
            for i in range(natm):
                for j in range(3):
                    self.atoms.positions[i, j] = positions0[i, j] + eps
                    grad_plus = np.array(self.run_kernel()["gradient"])
                    self.atoms.positions[i, j] = positions0[i, j] - eps
                    grad_minus = np.array(self.run_kernel()["gradient"])
                    self.atoms.positions[i, j] = positions0[i, j]
                    hessian[i, :, j, :] = (grad_plus - grad_minus) / (2.0 * eps / Bohr)
        finally:
            self.atoms.set_positions(positions0)
        return 0.5 * (hessian + hessian.transpose(1, 0, 3, 2))
