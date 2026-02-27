from typing import Optional, List, Literal
from types import SimpleNamespace

import numpy as np
from ase import Atoms
import ase.calculators.calculator
from ase.units import Hartree, Bohr, kB
from tblite.interface import Calculator, Result
from tblite.ase import TBLite, _create_api_calculator
from tblite.exceptions import TBLiteRuntimeError
from pyscf import gto

from .base_driver import BaseDriver


def heating_annealing(
    xtb: Calculator,
    temp: float,
    max_temp: float = 5000.0,
    heating_step: float = 300.0,
    annealing_step: float = 50.0,
) -> Optional[Result]:
    """
    Resolves xTB convergence issues using an electronic annealing strategy.

    Physical Context:
    For systems with small HOMO-LUMO gaps or strong electronic correlation, SCF 
    iterations often oscillate. Increasing the electronic temperature (Fermi smearing) 
    broadens the orbital occupancy, which smoothens the energy landscape and aids 
    convergence. This function recovers the target state by first finding a high-temperature 
    converged state and then gradually "cooling" the system using the previous 
    wavefunction as an initial guess (restart).

    Algorithm Logic:
    1. Linear Escalation (Heating Phase):
       The algorithm searches upwards from the target temperature using `heating_step` 
       until the first converged state is found at `temp_high`. If `max_temp` is 
       reached without success, the process terminates.

    2. Adaptive Bisection Descent (Annealing Phase):
       Using `temp_high` as a "safe anchor," the algorithm attempts to lower the 
       temperature toward the target. 
       - On Success: The anchor `temp_high` is updated to the current temperature, 
         and the search continues further down.
       - On Failure: The algorithm backtracks to the midpoint between the current 
         failed temperature and the last successful anchor (`temp_high`).
       - Convergence: The loop terminates successfully if the target temperature 
         is reached, or fails if the bisection gap becomes smaller than `annealing_step`.

    Args:
        xtb: A Calculator object supporting `singlepoint()` and `set()` methods.
        temp: The target physical temperature in Kelvin.
        max_temp: Maximum allowed electronic temperature in Kelvin.
        heating_step: Incremental step size for the initial upward scan.
        annealing_step: Minimum temperature resolution for the bisection search.

    Returns:
        Optional[Result]: The converged Result object at the lowest reachable 
        temperature (ideally `temp`), or None if no convergence was found.
    """
    kB_Hartree = kB / Hartree
    # escalate temperature until convergence or max_temp is reached
    temp_high = None
    conv_res = None
    for _t in np.arange(temp+heating_step, max_temp+heating_step, heating_step):
        t = _t.item()
        xtb.set("temperature", t * kB_Hartree)
        try:
            res = xtb.singlepoint()
            print(f"Heating: converged at {t:.2f} K.")
            conv_res = res
            temp_high = t
            break
        except TBLiteRuntimeError:
            print(f"Heating: failed to converge at {t:.2f} K.")
            continue
    
    if conv_res is None:
        print(f"Failed to converge even at {max_temp:.2f} K.")
        return None

    if abs(temp_high - temp) < 1e-3:
        return conv_res

    # descend temperature until orginal temp is reached and converged
    temp_try = temp
    while True:
        if abs(temp_try - temp_high) < annealing_step and temp_try > temp:
            print(f"Annealing limit reached. Best stable temperature: {temp_high:.2f} K.")
            return None
        xtb.set("temperature", temp_try * kB_Hartree)
        try:
            res = xtb.singlepoint(conv_res)
            print(f"Annealing: converged at {temp_try:.2f} K.")
            conv_res = res
            temp_high = temp_try
            # if temperature is close enough to target temp, consider it converged
            if abs(temp_try - temp) < 1e-3:
                return conv_res
            # descend temperature
            next_temp = (temp_try + temp) / 2
            if abs(next_temp - temp_try) < annealing_step:
                next_temp = temp
            temp_try = next_temp
        except TBLiteRuntimeError:
            # failed to converge, need to increase temperature
            print(f"Annealing: failed to converge at {temp_try:.2f} K.")
            next_temp = (temp_try + temp_high) / 2
            if abs(next_temp - temp_high) < annealing_step:
                print(f"Annealing limit reached. Best stable temperature: {temp_high:.2f} K.")
                return None
            temp_try = next_temp


class TBLiteCalculator(TBLite):
    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        try_annealing: bool = False,
        annealing_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(atoms, **kwargs)
        self.try_annealing = try_annealing
        self.annealing_config = {
            "max_temp": 5000.0,
            "heating_step": 300.0,
            "annealing_step": 50.0,
        }
        if annealing_config is not None:
            self.annealing_config.update(annealing_config)
        
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = ase.calculators.calculator.all_changes,
    ) -> None:
        """
        Add support for heating-annealing procedure if the initial calculation fails to converge.
        """

        if not properties:
            properties = ["energy"]
        ase.calculators.calculator.Calculator.calculate(self, atoms, properties, system_changes)

        self._check_api_calculator(system_changes)

        if self._xtb is None:
            self._xtb = _create_api_calculator(self.atoms, self.parameters)

        try:
            self._res = self._xtb.singlepoint(self._res)
            converged = True
        except RuntimeError as e:
            print(e)
            self._res = None
            converged = False
        if not converged and self.try_annealing:
            eTemp = self.parameters.electronic_temperature
            res = heating_annealing(
                self._xtb,
                temp=eTemp * Hartree / kB,
                max_temp=self.annealing_config["max_temp"],
                heating_step=self.annealing_config["heating_step"],
                annealing_step=self.annealing_config["annealing_step"],
            )
            if res is not None:
                self._res = res
                converged = True
        if not converged:
            raise RuntimeError("xTB calculation failed to converge.")

        # These properties are garanteed to exist for all implemented calculators
        self.results["energy"] = self._res["energy"] * Hartree
        self.results["energies"] = self._res["energies"] * Hartree
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -self._res["gradient"] * Hartree / Bohr
        self.results["charges"] = self._res["charges"]
        self.results["dipole"] = self._res["dipole"] * Bohr
        self.results["bond-orders"] = self._res["bond-orders"]
        # stress tensor is only returned for periodic systems
        if self.atoms.pbc.any():
            _stress = self._res["virial"] * Hartree / self.atoms.get_volume()
            self.results["stress"] = _stress.flat[[0, 4, 8, 5, 2, 1]]


class TBLiteDriver(BaseDriver):
    def __init__(
        self,
        atoms: Atoms,
        config: dict,
    ):
        super().__init__(atoms=atoms, config=config)
        self.calc: TBLiteCalculator = None
        self.xtb: Calculator = None
        self._res_cache: Result = None
        self.try_annealing = self.config.get("try_annealing", False)

    def build_calc(self, atoms: Optional[Atoms] = None) -> TBLiteCalculator:
        self.update_atoms(atoms)
        method = self.config.get("method", "GFN2-xTB")
        charge = self.atoms.info.get("charge", 0)
        multiplicity = self.atoms.info.get("multiplicity", 1)
        accuracy = self.config.get("accuracy", 1.0)
        guess = self.config.get("guess", "sad")
        eTemp = self.config.get("eTemp", 298.15)
        max_iter = self.config.get("max_iter", 250)
        mixer_damping = self.config.get("mixer_damping", 0.4)
        electric_field = self.config.get("electric_field", None)
        if electric_field is not None:
            electric_field = np.asarray(electric_field)
        spin_polarization = self.config.get("spin_polarization", None)
        solvation: dict = self.config.get("solvation", None)
        if solvation is not None:
            solv_method = solvation["method"].lower()
            if solv_method in {"alpb", "gbsa"}:
                solvent = solvation.get("solvent", "water")
                solution_state = solvation.get("solution_state", "gsolv")
                solv_args = (solv_method, (solvent, solution_state))
            elif solv_method == "cpcm":
                epsilon = solvation.get("epsilon", 78.3553)
                solv_args = (solv_method, epsilon)
            elif solv_method in {"gbe", "gb"}:
                epsilon = solvation.get("epsilon", 78.3553)
                born_kernel = solvation.get("born_kernel", "still")
                solv_args = (solv_method, (epsilon, born_kernel))
            else:
                raise ValueError(f"Unsupported solvation method: {solv_method}")
        else:
            solv_args = None
        verbosity = self.config.get("verbosity", 0)
        calc = TBLiteCalculator(
            method=method,
            charge=charge,
            multiplicity=multiplicity,
            accuracy=accuracy,
            guess=guess,
            electronic_temperature=eTemp,
            max_iter=max_iter,
            mixer_damping=mixer_damping,
            electric_field=electric_field,
            spin_polarization=spin_polarization,
            solvation=solv_args,
            verbosity=verbosity,
        )
        return calc

    def build_xtb(self, atoms: Optional[Atoms] = None) -> Calculator:
        self.update_atoms(atoms)
        method = self.config.get("method", "GFN2-xTB")
        charge = self.atoms.info.get("charge", 0)
        uhf = self.atoms.info.get("multiplicity", 1) - 1
        numbers = self.atoms.get_atomic_numbers()
        positions = self.atoms.get_positions() / Bohr
        periodic = self.atoms.get_pbc()
        lattice = self.atoms.get_cell().array / Bohr if periodic.any() else None
        xtb = Calculator(
            method=method,
            numbers=numbers,
            positions=positions,
            charge=charge,
            uhf=uhf,
            lattice=lattice,
            periodic=periodic,
        )
        xtb.set("accuracy", self.config.get("accuracy", 1.0))
        xtb.set("temperature", self.config.get("eTemp", 298.15) * kB / Hartree)
        xtb.set("max-iter", self.config.get("max_iter", 250))
        xtb.set("mixer-damping", self.config.get("mixer_damping", 0.4))
        xtb.set("verbosity", self.config.get("verbosity", 0))
        # external fields and spin polarization
        electric_field = self.config.get("electric_field", None)
        if electric_field is not None:
            xtb.add("electric-field", np.asarray(electric_field))
        spin_polarization = self.config.get("spin_polarization", None)
        if spin_polarization is not None:
            xtb.add("spin-polarization", spin_polarization)
        # solvation
        solvation: dict = self.config.get("solvation", None)
        if solvation is not None:
            solv_method = solvation["method"].lower()
            if solv_method in {"alpb", "gbsa"}:
                solvent = solvation.get("solvent", "water")
                solution_state = solvation.get("solution_state", "gsolv")
                xtb.add(f"{solv_method}-solvation", (solvent, solution_state))
            elif solv_method == "cpcm":
                epsilon = solvation.get("epsilon", 78.3553)
                xtb.add("cpcm-solvation", epsilon)
            elif solv_method in {"gbe", "gb"}:
                epsilon = solvation.get("epsilon", 78.3553)
                born_kernel = solvation.get("born_kernel", "still")
                xtb.add(f"{solv_method}-solvation", (epsilon, born_kernel))
            else:
                raise ValueError(f"Unsupported solvation method: {solv_method}")
        return xtb

    def to_ase_calc(self) -> TBLiteCalculator:
        if self.calc is None:
            self.calc = self.build_calc()
        return self.calc

    def to_pyscf_mf(self):
        mol = gto.M(
            atom=[(symb, coord) for symb, coord in zip(self.atoms.get_chemical_symbols(), self.atoms.get_positions())],
            charge=self.atoms.info.get("charge", 0),
            spin=self.atoms.info.get("multiplicity", 1) - 1,
        )
        if self.xtb is None:
            self.xtb = self.build_xtb()
        res = self.run_kernel(use_cache=True)
        e_tot = res["energy"] if res is not None else None
        dummy_mf = SimpleNamespace(mol=mol, e_tot=e_tot)
        return dummy_mf

    def run_kernel(
        self, atoms: Optional[Atoms] = None, use_cache: bool = True) -> Result:
        self.update_atoms(atoms)
        if self.xtb is None:
            self.xtb = self.build_xtb()
        else:
            positions = self.atoms.get_positions() / Bohr
            lattice = self.atoms.get_cell().array / Bohr if self.atoms.get_pbc().any() else None
            self.xtb.update(
                positions=positions,
                lattice=lattice,
            )
        res = self._res_cache if use_cache else None
        try:
            res = self.xtb.singlepoint(res=res)
            converged = True
        except TBLiteRuntimeError as e:
            print(e)
            res = None
            converged = False
        if not converged and self.try_annealing:
            print("Initial calculation failed to converge. Starting heating-annealing procedure...")
            eTemp = self.config.get("eTemp", 298.15)
            annealing_config = self.config.get("annealing", {})
            max_temp = annealing_config.get("max_temp", 5000.0)
            heating_step = annealing_config.get("heating_step", 300.0)
            annealing_step = annealing_config.get("annealing_step", 50.0)
            res = heating_annealing(self.xtb, temp=eTemp, max_temp=max_temp, heating_step=heating_step, annealing_step=annealing_step)
            converged = res is not None
        if use_cache and converged:
            self._res_cache = res
        return res
        
    def compute_energy(self, atoms: Optional[Atoms] = None) -> float:
        res = self.run_kernel(atoms)
        if res is None:
            print("Failed to converge")
            return None
        e_tot = res["energy"]  # in Hartree
        e_tot_eV = e_tot * Hartree  # convert from Hartree to eV
        print(f"Total Energy        [eV]: {e_tot_eV:16.10f}")
        print(f"Total Energy        [Eh]: {e_tot:16.10f}")
        return e_tot_eV  # return energy in eV

    def _compute_hessian_impl(
        self,
        atoms: Optional[Atoms],
        hess_format: Literal["pyscf", "ase"] = "ase",
    ) -> np.ndarray:
        self.update_atoms(atoms)
        natm = len(self.atoms)
        hessian = np.zeros((natm, natm, 3, 3))  # pyscf format
        eps = self.config.get("finite_diff_eps", 5e-3)
        for i in range(natm):
            for j in range(3):
                self.atoms.positions[i, j] += eps
                res_plus = self.run_kernel(use_cache=True)
                grad_plus = res_plus["gradient"]
                self.atoms.positions[i, j] -= 2 * eps
                res_minus = self.run_kernel(use_cache=True)
                grad_minus = res_minus["gradient"]
                hessian[i, :, j, :] = (grad_plus - grad_minus) / (2 * eps)
                self.atoms.positions[i, j] += eps
        hessian = self._convert_hessian_format(hessian=hessian, hess_format=hess_format)
        return hessian
