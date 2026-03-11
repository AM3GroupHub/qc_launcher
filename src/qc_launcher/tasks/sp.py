import time

from qc_launcher.drivers import BaseDriver


def run_sp(driver: BaseDriver) -> dict:
    start_time = time.time()
    energy = driver.compute_energy()
    end_time = time.time()
    print(f"Single point energy calculation completed in {end_time - start_time:.2f} seconds.\n")
    results = {"energy": (energy, "eV")}
    results.update(driver.dump_extra_results())
    return results


def run_grad(driver: BaseDriver) -> dict:
    start_time = time.time()
    forces = driver.compute_forces()
    end_time = time.time()
    print(f"Gradient calculation completed in {end_time - start_time:.2f} seconds.\n")
    results = {"forces": (forces, "eV/Å")}
    return results
