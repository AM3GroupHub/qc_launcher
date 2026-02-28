import os
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
    filename = os.path.splitext(os.path.basename(inputfile))[0]
    
    # initialize driver
    driver_config = config["driver"]
    driver_name = driver_config.pop("name")
    driver = get_driver(driver_name, atoms, driver_config)
    # load md configuration and run
    md_config = config["md"]
    run_md(driver, md_config, filename)

if __name__ == "__main__":
    main()
