import os
import argparse
import json

import yaml
import ase.io

from qc_launcher.drivers import get_driver
from qc_launcher.tasks import run_sp, run_grad, run_opt, run_freq, run_irc
from qc_launcher.utils.utils import to_jsonable

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

    driver_config: dict = config["driver"]
    driver_name: str = driver_config.pop("name")
    driver = get_driver(driver_name, atoms, driver_config)

    # initialize results dictionary
    results = {}

    # task 1: optimization
    if "opt" in config:
        opt_config: dict = config["opt"]
        opt_results = run_opt(driver, opt_config, filename)
        results.update(opt_results)
    
    # task 2: single point
    sp_results = run_sp(driver)
    results.update(sp_results)

    if "forces" in config and config["forces"]:
        grad_results = run_grad(driver)
        results.update(grad_results)

    # task 3: frequency
    if "freq" in config:
        freq_config: dict = config["freq"]
        freq_results = run_freq(driver, freq_config, filename)
        results.update(freq_results)

    # task 4: IRC
    if "irc" in config:
        irc_config: dict = config["irc"]
        irc_results = run_irc(driver, irc_config, filename)
        results.update(irc_results)

    save_json = config.get("save_json", False)
    if save_json:
        json_output = config.get("json_output", f"{filename}_db.json")
        system = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(),
            "charge": atoms.info.get("charge", 0),
            "multiplicity": atoms.info.get("multiplicity", 1),
        }
        json_dict = {
            "system": system,
            "config": config,
            "results": results,
        }
        with open(json_output, "w") as f:
            json.dump(to_jsonable(json_dict), f, indent=4)

if __name__ == "__main__":
    main()
