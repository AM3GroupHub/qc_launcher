import time

from qc_launcher.drivers import BaseDriver


def run_sp(driver: BaseDriver) -> None:
    start_time = time.time()
    driver.compute_energy()
    end_time = time.time()
    print(f"Single point energy calculation completed in {end_time - start_time:.2f} seconds.\n")


def run_grad(driver: BaseDriver) -> None:
    start_time = time.time()
    driver.compute_forces()
    end_time = time.time()
    print(f"Gradient calculation completed in {end_time - start_time:.2f} seconds.\n")
