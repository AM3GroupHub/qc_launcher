from qc_launcher.drivers import BaseDriver

from .common import timed


def run_sp(driver: BaseDriver) -> None:
    with timed("Single point energy calculation"):
        driver.compute_energy()


def run_grad(driver: BaseDriver) -> None:
    with timed("Gradient calculation"):
        driver.compute_forces()
