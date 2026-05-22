# QC Launcher

Lean CLI for quantum chemistry, GSM, NEB, MD, and `pysisyphus` workflows built around ASE-based drivers.

## Installation

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[gpu4pyscf]        # GPU acceleration for PySCF
pip install -e .[mace]             # MACE driver
pip install -e .[fairchem]         # UMA / FAIRChem driver
pip install -e .[gsm]              # installs forked pygsm from X1X1010/pygsm
pip install -e .[pysisyphus]       # pysisyphus bridge
pip install -e .[gsm,pysisyphus]   # both reaction-path extras
```

`[gsm]` installs:

```text
pygsm @ git+https://github.com/X1X1010/pygsm.git
```

Base package stays usable without `pygsm` or `pysisyphus`. Those dependencies are only required when the corresponding commands are used.

## CLI

Install gives one console entrypoint:

```bash
qc_launch <subcommand> ...
```

Available subcommands:

- `qc`: single-point, gradients, optimization, frequencies, IRC
- `md`: molecular dynamics
- `gsm`: growing string method
- `neb`: nudged elastic band
- `pysis`: run a `pysisyphus` YAML through the `qc_launcher` calculator bridge

## Quick Start

Single point:

```bash
(cd examples/single_point && qc_launch qc config.yaml)
```

Molecular dynamics:

```bash
(cd examples/6-md && qc_launch md 2-langevin.yaml)
```

Growing string method:

```bash
(cd examples/5-gsm && qc_launch gsm 1-gsm.yaml)
```

`pysisyphus` bridge:

```bash
(cd examples/optts_pysisyphus && qc_launch pysis config.yaml)
```


## Examples

Placeholder inputs live under [`examples/`](/home/admin/storage/repo/qc_launcher/examples). They are meant to be edited, not run as-is for production work.

- [`examples/single_point/`](/home/admin/storage/repo/qc_launcher/examples/single_point): single-point energy
- [`examples/opt/`](/home/admin/storage/repo/qc_launcher/examples/opt): Sella minimum optimization
- [`examples/freq/`](/home/admin/storage/repo/qc_launcher/examples/freq): PySCF frequency analysis
- [`examples/optts_sella/`](/home/admin/storage/repo/qc_launcher/examples/optts_sella): Sella transition-state optimization
- [`examples/irc_sella/`](/home/admin/storage/repo/qc_launcher/examples/irc_sella): Sella IRC
- [`examples/4-neb/`](/home/admin/storage/repo/qc_launcher/examples/4-neb): NEB
- [`examples/5-gsm/`](/home/admin/storage/repo/qc_launcher/examples/5-gsm): GSM
- [`examples/6-md/`](/home/admin/storage/repo/qc_launcher/examples/6-md): MD
- [`examples/optts_pysisyphus/`](/home/admin/storage/repo/qc_launcher/examples/optts_pysisyphus): `pysisyphus` OptTS
- [`examples/irc_pysisyphus/`](/home/admin/storage/repo/qc_launcher/examples/irc_pysisyphus): `pysisyphus` IRC

Replace the shipped XYZ placeholders with your own structures before running.

## Config Templates

Reusable templates are in [`config/`](/home/admin/storage/repo/qc_launcher/config). They point at the dummy structures in `examples/` and can be run from the repository root, for example:

```bash
qc_launch qc config/run_single_point.yaml
qc_launch neb config/run_neb.yaml
qc_launch pysis config/run_optts_pysisyphus.yaml
```

## Notes

- GSM requires the forked `pygsm` package from `X1X1010/pygsm`.
- `qc_launch gsm` raises a clear install hint if `[gsm]` is missing.
- `qc_launch pysis` raises a clear install hint if `[pysisyphus]` is missing.
- Any ASE-compatible driver in `qc_launcher` that provides energies and forces can be used for GSM.
