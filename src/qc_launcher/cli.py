from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import ase.io
import yaml
from ase import Atoms

from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency
from qc_launcher.drivers import get_driver


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{config_path}' must contain a top-level mapping.")
    return config


def apply_charge_and_multiplicity(
    atoms: Atoms | list[Atoms],
    charge: int | None,
    multiplicity: int | None,
) -> None:
    if isinstance(atoms, list):
        atoms_list = atoms
    else:
        atoms_list = [atoms]
    for frame in atoms_list:
        if charge is not None:
            frame.info["charge"] = charge
        if multiplicity is not None:
            frame.info["multiplicity"] = multiplicity


def load_single_structure(config: dict[str, Any]) -> tuple[Atoms, str]:
    inputfile = config["inputfile"]
    atoms = ase.io.read(inputfile)
    apply_charge_and_multiplicity(
        atoms,
        config.get("charge"),
        config.get("multiplicity"),
    )
    return atoms, Path(inputfile).stem


def load_multi_structure(config: dict[str, Any]) -> tuple[list[Atoms], str]:
    inputfile = config["inputfile"]
    atoms_list = list(ase.io.read(inputfile, index=":"))
    apply_charge_and_multiplicity(
        atoms_list,
        config.get("charge"),
        config.get("multiplicity"),
    )
    return atoms_list, Path(inputfile).stem


def build_driver(
    config: dict[str, Any],
    atoms: Atoms | None,
    *,
    verbose: bool = True,
):
    driver_config = dict(config["driver"])
    driver_name = driver_config.pop("name")
    return get_driver(driver_name, atoms, driver_config, verbose=verbose)


def normalize_task_config(config: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    if task_name not in config:
        return None
    task_setting = config[task_name]
    if isinstance(task_setting, dict):
        return dict(task_setting)
    if isinstance(task_setting, bool) and task_setting:
        return {}
    raise ValueError(
        f"Invalid {task_name} configuration. Expected a mapping or `true`."
    )


def load_opt_runner() -> Callable[..., dict[str, Any]]:
    try:
        from qc_launcher.tasks.opt import run_opt
    except ModuleNotFoundError as exc:
        if is_missing_package(exc, "sella"):
            raise ModuleNotFoundError(
                "Optimization tasks require `sella`. Install it in the runtime environment."
            ) from exc
        raise
    return run_opt


def load_irc_runner() -> Callable[..., dict[str, Any]]:
    try:
        from qc_launcher.tasks.irc import run_irc
    except ModuleNotFoundError as exc:
        if is_missing_package(exc, "sella"):
            raise ModuleNotFoundError(
                "IRC tasks require `sella`. Install it in the runtime environment."
            ) from exc
        raise
    return run_irc

def load_gsm_runner() -> Callable[..., dict[str, Any]]:
    try:
        from qc_launcher.tasks.gsm import run_gsm
    except ModuleNotFoundError as exc:
        if is_missing_package(exc, "pygsm"):
            raise ModuleNotFoundError(
                "GSM tasks require `pygsm`. Install it in the runtime environment."
            ) from exc
        raise
    return run_gsm


def launch_qc(config_path: str) -> None:
    from qc_launcher.tasks.freq import run_freq
    from qc_launcher.tasks.sp import run_grad, run_sp

    config = load_yaml_config(config_path)
    atoms, filename = load_single_structure(config)
    driver = build_driver(config, atoms)

    opt_config = normalize_task_config(config, "opt")
    if opt_config is not None:
        run_opt = load_opt_runner()
        run_opt(driver, opt_config, filename)

    run_sp(driver)

    if "forces" in config:
        if config["forces"] is not True:
            raise ValueError("`forces` must be set to `true` to compute gradients.")
        run_grad(driver)

    freq_config = normalize_task_config(config, "freq")
    if freq_config is not None:
        run_freq(driver, freq_config, filename)

    irc_config = normalize_task_config(config, "irc")
    if irc_config is not None:
        run_irc = load_irc_runner()
        run_irc(driver, irc_config, filename)


def launch_md(config_path: str) -> None:
    from qc_launcher.tasks.md import run_md

    config = load_yaml_config(config_path)
    atoms, filename = load_single_structure(config)
    driver = build_driver(config, atoms)
    md_config = normalize_task_config(config, "md")
    if md_config is None:
        raise ValueError("MD run requires an `md` configuration block.")
    run_md(driver, md_config, filename)


def launch_gsm(config_path: str) -> None:
    run_gsm = load_gsm_runner()
    config = load_yaml_config(config_path)
    atoms_list, filename = load_multi_structure(config)
    if len(atoms_list) == 0:
        raise ValueError("GSM run requires at least one input geometry.")
    driver = build_driver(config, atoms_list[0], verbose=False)
    gsm_config = normalize_task_config(config, "gsm")
    if gsm_config is None:
        raise ValueError("GSM run requires a `gsm` configuration block.")
    start_time = time.time()
    run_gsm(driver, gsm_config, atoms_list, filename)
    end_time = time.time()
    print(f"GSM calculation completed in {end_time - start_time:.2f} seconds.\n")


def launch_neb(config_path: str) -> None:
    from qc_launcher.tasks.neb import run_neb

    config = load_yaml_config(config_path)
    atoms_list, filename = load_multi_structure(config)
    if len(atoms_list) == 0:
        raise ValueError("NEB run requires at least one input geometry.")
    driver = build_driver(config, atoms_list[0])
    neb_config = normalize_task_config(config, "neb")
    if neb_config is None:
        raise ValueError("NEB run requires a `neb` configuration block.")
    start_time = time.time()
    run_neb(driver, neb_config, atoms_list, filename)
    end_time = time.time()
    print(f"NEB calculation completed in {end_time - start_time:.2f} seconds.\n")


def launch_autoneb(config_path: str) -> None:
    from qc_launcher.tasks.autoneb import run_autoneb

    config = load_yaml_config(config_path)
    atoms_list, filename = load_multi_structure(config)
    if len(atoms_list) != 2:
        raise ValueError("AutoNEB run requires exactly two input geometries.")
    driver = build_driver(config, atoms_list[0])
    autoneb_config = normalize_task_config(config, "autoneb")
    if autoneb_config is None:
        raise ValueError("AutoNEB run requires an `autoneb` configuration block.")
    start_time = time.time()
    run_autoneb(driver, autoneb_config, atoms_list, filename)
    end_time = time.time()
    print(f"AutoNEB calculation completed in {end_time - start_time:.2f} seconds.\n")


def launch_pysis(config_path: str) -> None:
    try:
        import pysisyphus.run as pysis_run
        from pysisyphus.run import load_run_dict, run_from_dict
    except ModuleNotFoundError as exc:
        if is_missing_package(exc, "pysisyphus"):
            raise missing_optional_dependency("pysisyphus interface", "pysisyphus") from exc
        raise

    from qc_launcher.drivers import QCSisyphusCalc

    pysis_run.CALC_DICT["qc_launcher"] = QCSisyphusCalc

    run_dict = load_run_dict(config_path)
    run_from_dict(run_dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qc_launch",
        description="Unified CLI for qc_launcher workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    qc_parser = subparsers.add_parser("qc", help="Run QC workflow.")
    qc_parser.add_argument("config", help="Path to YAML configuration file.")
    qc_parser.set_defaults(handler=lambda args: launch_qc(args.config))

    md_parser = subparsers.add_parser("md", help="Run MD workflow.")
    md_parser.add_argument("config", help="Path to YAML configuration file.")
    md_parser.set_defaults(handler=lambda args: launch_md(args.config))

    gsm_parser = subparsers.add_parser("gsm", help="Run GSM workflow.")
    gsm_parser.add_argument("config", help="Path to YAML configuration file.")
    gsm_parser.set_defaults(handler=lambda args: launch_gsm(args.config))

    neb_parser = subparsers.add_parser("neb", help="Run NEB workflow.")
    neb_parser.add_argument("config", help="Path to YAML configuration file.")
    neb_parser.set_defaults(handler=lambda args: launch_neb(args.config))

    autoneb_parser = subparsers.add_parser("autoneb", help="Run AutoNEB workflow.")
    autoneb_parser.add_argument("config", help="Path to YAML configuration file.")
    autoneb_parser.set_defaults(handler=lambda args: launch_autoneb(args.config))
    
    pysis_parser = subparsers.add_parser("pysis", help="Pass arguments through to pysisyphus.")
    pysis_parser.add_argument("config", help="Path to YAML configuration file.")
    pysis_parser.set_defaults(handler=lambda args: launch_pysis(args.config))

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv_list = list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    args = parser.parse_args(argv_list)
    args.handler(args)
