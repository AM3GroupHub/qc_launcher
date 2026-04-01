import os
import argparse
import time

import yaml
import ase.io

from qc_launcher.drivers import get_driver
from qc_launcher.tasks import run_gsm


def main():
    parser = argparse.ArgumentParser(description="Launch a GSM task with a YAML configuration file.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: dict = yaml.safe_load(f)

    inputfile = config["inputfile"]
    charge = config.get("charge", None)
    multiplicity = config.get("multiplicity", None)
    atoms_list = ase.io.read(inputfile, index=":")
    if charge is not None:
        for atoms in atoms_list:
            atoms.info["charge"] = charge
    if multiplicity is not None:
        for atoms in atoms_list:
            atoms.info["multiplicity"] = multiplicity
    filename = os.path.splitext(os.path.basename(inputfile))[0]

    driver_config = dict(config["driver"])
    driver_name: str = driver_config.pop("name")
    driver = get_driver(driver_name, atoms_list[0], driver_config, verbose=False)

    gsm_config = dict(config["gsm"])
    start_time = time.time()
    run_gsm(driver, gsm_config, atoms_list, filename)
    end_time = time.time()
    print(f"GSM calculation completed in {end_time - start_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
