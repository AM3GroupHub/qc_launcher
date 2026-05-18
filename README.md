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
- `pysis`: pass-through runner for `pysisyphus`

## Quick Start

Quantum chemistry:

```bash
qc_launch qc examples/qc/pyscf_qc.yaml
```

Molecular dynamics:

```bash
qc_launch md examples/md/tblite_md.yaml
```

Growing string method:

```bash
qc_launch gsm examples/gsm/pyscf_gsm.yaml
```

`pysisyphus` bridge:

```bash
qc_launch pysis --yaml examples/pysisyphus/RUN.yaml
```


## Examples

Placeholder inputs live under [`examples/`](/home/admin/storage/repo/qc_launcher/examples). They are meant to be edited, not run as-is for production work.

- [`examples/qc/`](/home/admin/storage/repo/qc_launcher/examples/qc): PySCF and TBLite QC examples
- [`examples/md/`](/home/admin/storage/repo/qc_launcher/examples/md): short MD example
- [`examples/gsm/`](/home/admin/storage/repo/qc_launcher/examples/gsm): GSM example with two-frame XYZ placeholder
- [`examples/pysisyphus/`](/home/admin/storage/repo/qc_launcher/examples/pysisyphus): `pysisyphus` TS optimization template

Replace the shipped XYZ placeholders with your own structures before running.

## Notes

- GSM requires the forked `pygsm` package from `X1X1010/pygsm`.
- `qc_launch gsm` raises a clear install hint if `[gsm]` is missing.
- `qc_launch pysis` raises a clear install hint if `[pysisyphus]` is missing.
- Any ASE-compatible driver in `qc_launcher` that provides energies and forces can be used for GSM.

