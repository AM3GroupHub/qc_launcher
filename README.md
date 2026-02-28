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

## Supported Drivers

- **pyscf**: Ab initio electronic structure calculations
- **mace**: Machine learning interatomic potentials
- **tblite**: Tight-binding extended tight-binding calculations
- **uma**: Universal machine learning potentials

## Supported Tasks

- Geometry optimization (min/TS)
- Frequency/vibrational analysis
- Intrinsic reaction coordinate (IRC)
- Molecular dynamics (NVE/NVT/NPT)
- Nudged elastic band (NEB)

## Configuration

See [config/](config/) directory for example YAML files demonstrating task configurations.
