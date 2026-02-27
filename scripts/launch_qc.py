import argparse

import yaml
import ase.io

from qc_launcher.drivers import get_driver
from qc_launcher.tasks import run_opt, run_freq, run_irc


def main():
    parser = argparse.ArgumentParser(description="Launch a script with a YAML configuration file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: dict = yaml.safe_load(f)

    # read inputfile
    inputfile = config["inputfile"]
    charge = config.get("charge", None)
    multiplicity = config.get("multiplicity", None)
    atoms = ase.io.read(inputfile)
    if charge is not None:
        atoms.info["charge"] = charge
    if multiplicity is not None:
        atoms.info["multiplicity"] = multiplicity

    driver_config: dict = config["driver"]
    driver_name: str = driver_config.pop("name")
    driver = get_driver(driver_name, driver_config)

    # task 1: optimization
    if "opt" in config:
        opt_config: dict = config["opt"]
        run_opt(atoms, driver, opt_config)
    
    # task 2: single point
    driver.compute_energy()

    # task 3: frequency
    if "freq" in config:
        freq_config: dict = config["freq"]
        run_freq(atoms, driver, freq_config)

    # task 4: IRC
    if "irc" in config:
        irc_config: dict = config["irc"]
        run_irc(atoms, driver, irc_config)


if __name__ == "__main__":
    main()