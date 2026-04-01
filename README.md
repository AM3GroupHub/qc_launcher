# QC Launcher

A Python interface for quantum chemistry and molecular dynamics calculations using PySCF, Sella and ASE


## Installation

```bash
pip install -e .
```

Optional dependencies:
```bash
pip install -e .[gpu4pyscf]  # GPU acceleration for PySCF
pip install -e .[mace]        # MACE-based potentials
pip install -e .[fairchem]    # FAIRChem models
```

## Quick Start

### Quantum Chemistry Calculations

```bash
python scripts/launch_qc.py run_qc.yaml
```

Example configuration:
```yaml
inputfile: structure.xyz
charge: 0
multiplicity: 1

driver:
  name: pyscf
  method: b3lyp
  basis: 6-31g*

opt:
  fmax: 0.02
  max_steps: 150

freq:
  temperature: 298.15
```

### Molecular Dynamics

```bash
python scripts/launch_md.py config/run_md.yaml
```

### Growing String Method

```bash
python scripts/launch_gsm.py config/run_gsm.yaml
```

`launch_gsm.py` runs double-ended GSM through the direct `pysisyphus` API.

- Provide a two-frame input file containing reactant and product.
- GSM is limited to the string search itself. TS optimization and IRC should be run as separate tasks.
- Any qc_launcher driver that provides energies and forces can be used.
- Install the local `pysisyphus` checkout into the runtime environment before using GSM:

```bash
cd ../pysisyphus
pip install -e .
```

## Supported Drivers

- **pyscf**: Ab initio electronic structure calculations
- **mace**: Machine learning interatomic potentials
- **tblite**: Tight-binding extended tight-binding calculations
- **uma**: Universal machine learning potentials

## Supported Tasks

- Geometry optimization (min/TS)
- Frequency/vibrational analysis
- Intrinsic reaction coordinate (IRC)
- Growing string method (DE-GSM via `pysisyphus`)
- Molecular dynamics (NVE/NVT/NPT)
- Nudged elastic band (NEB)

## Configuration

See [config/](config/) directory for example YAML files demonstrating task configurations.
