"""Command-line entry point: ``qc_launch <subcommand> config.yaml``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Union

import ase.io
import yaml
from ase import Atoms

from qc_launcher.drivers import BaseDriver, driver_from_config
from qc_launcher.tasks import load_task
from qc_launcher.tasks.common import timed
from qc_launcher.utils.config import rename_deprecated_keys
from qc_launcher.utils.optional import is_missing_package, missing_optional_dependency

Structures = Union[Atoms, list[Atoms]]


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file '{config_path}' must contain a top-level mapping.")
    return config


def load_structures(config: dict[str, Any], *, multi: bool = False) -> tuple[Structures, str]:
    """Read ``inputfile`` (last frame, or every frame when ``multi``) and tag charge/multiplicity."""
    inputfile = config["inputfile"]
    structures = list(ase.io.read(inputfile, index=":")) if multi else ase.io.read(inputfile)
    for frame in structures if multi else [structures]:
        for key in ("charge", "multiplicity"):
            if config.get(key) is not None:
                frame.info[key] = config[key]
    return structures, Path(inputfile).stem


def normalize_task_config(config: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    """Return the task block as a dict (``true`` means an empty block), or None when absent."""
    if task_name not in config:
        return None
    task_setting = config[task_name]
    if isinstance(task_setting, dict):
        return rename_deprecated_keys(task_setting, task_name)
    if task_setting is True:
        return {}
    raise ValueError(f"Invalid {task_name} configuration. Expected a mapping or `true`.")


def require_task_config(config: dict[str, Any], task_name: str) -> dict[str, Any]:
    task_config = normalize_task_config(config, task_name)
    if task_config is None:
        raise ValueError(f"`qc_launch {task_name}` requires a `{task_name}` configuration block.")
    return task_config


def _prepare_run(
    config_path: str, *, multi: bool = False, verbose: bool = True
) -> tuple[dict[str, Any], Structures, str, BaseDriver]:
    config = load_yaml_config(config_path)
    structures, filename = load_structures(config, multi=multi)
    first = structures[0] if multi else structures
    driver = driver_from_config(config["driver"], first, verbose=verbose)
    return config, structures, filename, driver


def launch_qc(config_path: str) -> None:
    """opt -> single point -> (forces) -> freq -> irc, each enabled by its config block."""
    config, atoms, filename, driver = _prepare_run(config_path)

    opt_config = normalize_task_config(config, "opt")
    if opt_config is not None:
        load_task("opt")(driver, opt_config, filename)

    load_task("sp")(driver)

    if "forces" in config:
        if config["forces"] is not True:
            raise ValueError("`forces` must be set to `true` to compute gradients.")
        load_task("grad")(driver)

    freq_config = normalize_task_config(config, "freq")
    if freq_config is not None:
        load_task("freq")(driver, freq_config, filename)

    irc_config = normalize_task_config(config, "irc")
    if irc_config is not None:
        load_task("irc")(driver, irc_config, filename)


def launch_md(config_path: str) -> None:
    config, atoms, filename, driver = _prepare_run(config_path)
    load_task("md")(driver, require_task_config(config, "md"), filename)


def launch_path(task_name: str, config_path: str, *, verbose: bool = True) -> None:
    """Multi-structure tasks (gsm, neb, autoneb); the task validates the number of frames."""
    config, atoms_list, filename, driver = _prepare_run(config_path, multi=True, verbose=verbose)
    task_config = require_task_config(config, task_name)
    runner = load_task(task_name)
    with timed(f"{task_name.upper()} calculation"):
        runner(driver, task_config, atoms_list, filename)


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
    run_from_dict(load_run_dict(config_path))


_SUBCOMMANDS: dict[str, tuple[str, Callable[[str], None]]] = {
    "qc": ("Single point, forces, optimization, frequencies, IRC (`opt`, `forces`, `freq`, `irc` blocks).", launch_qc),
    "md": ("Molecular dynamics (`md` block).", launch_md),
    "gsm": (
        "Growing string method (`gsm` block; reactant and product frames).",
        partial(launch_path, "gsm", verbose=False),
    ),
    "neb": ("Nudged elastic band (`neb` block; two or more frames).", partial(launch_path, "neb")),
    "autoneb": ("ASE AutoNEB (`autoneb` block; exactly two frames).", partial(launch_path, "autoneb")),
    "pysis": ("Run a pysisyphus YAML through the qc_launcher calculator bridge.", launch_pysis),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qc_launch", description="Unified CLI for qc_launcher workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, handler) in _SUBCOMMANDS.items():
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("config", help="Path to YAML configuration file.")
        subparser.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    args.handler(args.config)
