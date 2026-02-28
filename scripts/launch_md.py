import argparse

import yaml
import ase.io

from qc_launcher.drivers import get_driver
from qc_launcher.tasks import run_md


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
    
    driver_config = config["driver"]
    driver_name = driver_config.pop("name")
    driver = get_driver(driver_name, driver_config)
    run_md(driver, config)

if __name__ == "__main__":
    main()