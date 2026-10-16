# QC Launcher

`qc_launch` is a small YAML-driven command line for routine computational chemistry on ASE-based
drivers: PySCF / gpu4pyscf (DFT, CPU or GPU), tblite (GFN1/GFN2-xTB), MACE and FAIRChem UMA (machine-learned
potentials). On top of the drivers it runs single points, Sella minimum / transition-state optimizations and
IRCs, harmonic frequencies with RRHO and quasi-RRHO thermochemistry, ASE NEB / AutoNEB, the growing string
method (pygsm), molecular dynamics, and a calculator bridge for pysisyphus.

## Installation

```bash
pip install -e .                    # ase, numpy, pyyaml, pyscf, sella
pip install -e .[gpu4pyscf]         # GPU acceleration for the pyscf driver (CUDA 12)
pip install -e .[tblite]            # tblite driver
pip install -e .[mace]              # MACE driver (mace-torch)
pip install -e .[fairchem]          # UMA driver (fairchem-core)
pip install -e .[gsm]               # growing string method (forked pygsm: X1X1010/pygsm)
pip install -e .[pysisyphus]        # pysisyphus bridge
pip install -e .[dev]               # pytest, ruff
```

Optional backends are imported only when a config asks for them; a missing one produces an error naming the
extra to install.

## Command line

```bash
qc_launch <subcommand> config.yaml
```

| Subcommand | Input frames | Blocks read from the YAML |
|---|---|---|
| `qc` | one (last frame of `inputfile`) | optional `opt`, then a single point, optional `forces: true`, `freq`, `irc`, in that order |
| `md` | one | `md` |
| `neb` | two (interpolated) or `num_images + 2` | `neb` |
| `autoneb` | exactly two | `autoneb` |
| `gsm` | two (reactant, product) | `gsm` |
| `pysis` | a pysisyphus YAML | handled by pysisyphus; use `calc: {type: qc_launcher, driver: {...}}` |

A task block may be `true` to use its defaults, for example `opt: true`.

Quick start:

```bash
cd examples/1-single_point && qc_launch qc 7-tblite.yaml
cd examples/3-transition_state && qc_launch qc 1-optts.yaml
cd examples/6-dynamics && qc_launch md 2-langevin.yaml
```

## Configuration

Top-level keys: `inputfile` (any format ASE reads), `charge` (0), `multiplicity` (1), `driver` (required, see
below), `forces: true` (print forces after the single point), and the task blocks.

### Drivers

`driver.name` selects the backend (case-insensitive). Keys and defaults:

**pyscf**

| Key | Default | Notes |
|---|---|---|
| `xc` | `B3LYP` | any libxc name; names ending in `3c` (`B97-3c`, `r2SCAN-3c`) select a composite method via gpu4pyscf, which then sets basis / ECP / dispersion |
| `basis`, `ecp` | `def2-SVP`, none | |
| `restricted` | none | `true` RKS/ROKS, `false` UKS, unset: by spin |
| `nlc`, `dispersion` | `''`, none | passed to the KS object |
| `grids`, `nlcgrids` | pyscf defaults | `{atom_grid: ...}` or `{level: n}` |
| `with_df`, `auxbasis` | `true`, none | density fitting |
| `with_gpu` | `true` | falls back to CPU when cupy is missing |
| `solvation` | none | `{model: IEF-PCM \| C-PCM \| SS(V)PE \| COSMO, epsilon: 78.4}` or `{model: ..., solvent: water}` (184 Gaussian solvents), or `{model: SMD, solvent: water}` / `{model: SMD, solvent_descriptors: [...]}` |
| `scf_conv_tol`, `direct_scf_tol` | `1e-8`, `1e-8` | |
| `scf_max_iter`, `level_shift`, `diis_space` | `50`, `0`, `8` | |
| `retry_soscf` | `false` | retry a non-converged SCF with second-order SCF |
| `max_unconverged_steps` | none | warn after this many non-converged SCFs in an optimization |
| `numerical_hessian`, `finite_diff_step` | `false`, `5e-3` Å | central differences of analytic gradients instead of the analytic Hessian |
| `verbosity`, `max_memory` (GB), `threads` | `2`, none, `OMP_NUM_THREADS` | |

**tblite**

| Key | Default | Notes |
|---|---|---|
| `method` | `GFN2-xTB` | or `GFN1-xTB` |
| `accuracy`, `guess`, `electronic_temperature` (K), `scf_max_iter`, `mixer_damping`, `verbosity` | `1.0`, `sad`, `298.15`, `250`, `0.4`, `0` | |
| `electric_field` (V/Å), `spin_polarization` | none | |
| `solvation` | none | `{model: alpb \| gbsa, solvent, solution_state}`, `{model: cpcm, epsilon}` or `{model: gbe \| gb, epsilon, born_kernel}` |
| `retry_annealing`, `annealing` | `false`, `{max_temp: 5000, heating_step: 300, annealing_step: 50}` | on SCC failure, converge at a higher electronic temperature and cool back down |
| `finite_diff_step` | `5e-3` Å | Hessians are central differences of the gradient |

SCC failures raise an error (after the annealing retry when enabled).

**mace**

| Key | Default | Notes |
|---|---|---|
| `model_path` | required | |
| `model_type` | `omol` | `omol` or `mp`; `mp` models also take `head`, `dispersion`, `dispersion_xc`, `damping` |
| `device`, `precision` | auto, `float64` | |
| `neighbor_list` | `nvalchemi` if nvalchemiops is installed, else `mace` | the nvalchemiops neighbour list is installed process-wide when the calculator is built |

Hessians come from MACE's analytic `get_hessian`.

**uma**

| Key | Default | Notes |
|---|---|---|
| `model_path` | required | `atom_refs_path` defaults to `iso_atom_elem_refs.yaml` next to the model |
| `task_name`, `device` | `omol`, auto | |
| `finite_diff_step`, `batch_size` | `5e-3` Å, `128` | Hessians are batched central differences of the predicted forces |

### Tasks

**opt** (Sella): `ts` (false), `calc_hess` (= `ts`), `internal` (true), `trajectory`, `outputfile`, `delta0`
(0.1), `eta` (1e-4), `gamma` (0.1), `nsteps_per_diag` (3), `diag_every_n`, `constraints`
(`{fix_translation: [i], fix_bond: [[i, j]], fix_angle: [[i, j, k]], fix_dihedral: [[i, j, k, l]]}`),
`constraints_tol` (1e-5), `max_steps` (150). Convergence follows Gaussian's criteria: `ediff` (eV), `fmax` /
`frms` (eV/Å, RMS over all 3N components), `dmax` / `drms` (Å).

**freq**: `temperature` (298.15 K), `pressure` (101325 Pa), `vibfile`, `save_hess` (false), `symm_geom_tol`
(Å, pyscf symmetry detection tolerance), `qrrho` (true), `alpha` (4), `omega0` (100 cm⁻¹), `bav` (1e-44 kg m²,
or `auto`). Writes `<stem>_vib.txt` (PyVibMS format) and `<stem>_thermo.json`. See *Thermochemistry* below.

**irc** (Sella): `trajectory`, `outputfile`, `ninner_iter` (10), `dx` (0.1 Å), `eta` (1e-4), `keep_going`
(false), `diag_every_n`, `fmax`, `max_steps` (10), `direction` (`forward` | `reverse` | `both`).

**md** (ASE): `ensemble` (`NVE`, `Langevin`, `NVT` = Nosé–Hoover chain, `NPT` / `MTKNPT`, `IsotropicMTKNPT`,
`MelchionnaNPT`); `minimize` (true, false, or a block `{optimizer: FIRE | FIRE2 | LBFGS, fmax, max_steps,
trajectory, kwargs}` for the pre-minimization); `timestep` (1 fs), `temperature` (300 K), `nsteps` (1000), `loginterval`
(100), `trajectory`, `logfile`, `append_trajectory`, `init_velocities`, `log_stress`, `seed`, `resume`.
Langevin: `friction` (0.002 fs⁻¹). NVT/NPT: `tdamp`, `tchain`, `tloop`; MTK NPT: `pressure` (Pa), `pdamp`,
`pchain`, `ploop`; Melchionna: `externalstress`, `mask`, `ttime`, `ptime`, `bulk_modulus` (GPa). Atoms with a
non-zero tag in the input are kept fixed. `resume: true` continues an existing trajectory for the remaining
steps.

**neb** (ASE NEB + FIRE): `num_images` (frames − 2), `fmax` (0.05), `climb` (false), `climb_after` (0),
`max_steps` (200), `endpoint_max_steps` (step limit of the endpoint relaxations), `spring_constant` (0.1),
`neb_method` (improvedtangent), `fixed_reactant` / `fixed_product`, `interpolate_method` (idpp), `trajectory`.

**autoneb** (ASE AutoNEB): as `neb` plus `max_images` (11), `space_energy_ratio` (0.5), `smooth_curve`
(false), `scratch_dir` (`scratch`), `clear_scratch` (false).

**gsm** (pygsm, double-ended; keys follow the pygsm fork's parameter names): `nnodes` (11),
`coordinate_type` (`TRIC` | `DLC` | `HDLC`), `optimizer_method` (`eigenvector_follow` | `lbfgs`), `linesearch`
(NoLineSearch), `ts_search` (`climb` | `exact` | `none`), `dmax` (0.1), `add_node_tol` (0.1), `conv_tol`
(5e-4), `conv_ediff` / `conv_gmax` (100 = disabled; pygsm's own defaults are 0.1 / 0.001), `max_iters` (100
GSM iterations), `opt_steps` (3 optimizer steps per GSM iteration), `endpoint_max_steps` (100),
`fixed_reactant` / `fixed_product`, `string_id` (0), `output_dir`, `log_level`.

**pysisyphus bridge**: in a pysisyphus YAML use

```yaml
calc:
  type: qc_launcher
  driver: {name: tblite, method: GFN2-xTB, scf_max_iter: 100}
  charge: 0
  mult: 1
```

and run it with `qc_launch pysis run.yaml`.

### Deprecated keys

Similar settings use the same name everywhere; the older spellings keep working and print a
`ConfigDeprecationWarning`:

| Block | Old | New |
|---|---|---|
| driver | `thread`, `verbose`, `scf_max_cycle` / `max_iter`, `disp`, `etemp`, `numerical_hess`, `finite_diff_eps` | `threads`, `verbosity`, `scf_max_iter`, `dispersion`, `electronic_temperature`, `numerical_hessian`, `finite_diff_step` |
| driver | `with_solvent: true` + `solvent: {method, eps, solvent_descriptor}` | `solvation: {model, epsilon, solvent_descriptors}` (the block alone enables solvation) |
| opt | `delta` | `delta0` |
| freq | `Bav` | `bav` |
| irc | `irc_trajectory`, `irc_output`, `steps`, `direction: backward` | `trajectory`, `outputfile`, `max_steps`, `direction: reverse` |
| md | `dtamp`; flat `optimizer`, `fmax`, `max_min_steps`, `min_trajectory`, `opt_kwargs` | `tdamp`; `minimize: {optimizer, fmax, max_steps, trajectory, kwargs}` |
| neb / autoneb | `max_neb_steps` / `steps`, `max_opt_steps` | `max_steps`, `endpoint_max_steps` |
| gsm | `num_nodes` / `max_nodes`, `max_gsm_steps` / `max_cycles`, `max_opt_steps`, `ID`, `fmax`, `ediff`, `line_search`, `endpoint_opt_steps`, `climb` | `nnodes`, `max_iters`, `opt_steps`, `string_id`, `conv_gmax`, `conv_ediff`, `linesearch`, `endpoint_max_steps`, `ts_search` |

## Outputs

| Task | Files (`<stem>` is the input file name without extension) |
|---|---|
| opt | `<stem>_opt.traj`, `<stem>_opt.xyz` |
| freq | `<stem>_vib.txt`, `<stem>_thermo.json`, optional `<stem>_hess.txt` (eV/Å², ASE layout) |
| irc | `<stem>_irc.traj`, `<stem>_irc.xyz` (reactant → TS → product) |
| md | `<stem>_md.traj`, `<stem>_min.traj`, log file |
| neb / autoneb | `<stem>_neb.xyz`, `<stem>_ts.xyz`, `<stem>_neb.traj` |
| gsm | `<stem>_gsm.xyz`, `<stem>_ts.xyz` |

## Thermochemistry

Frequencies come from pyscf's harmonic analysis of the driver's Hessian (translations and rotations projected
out; imaginary modes are reported and excluded from the thermochemistry). RRHO quantities are pyscf's at the
given temperature and pressure (1 atm default), with the symmetry number from pyscf's point-group detection.

With `qrrho: true` the vibrational entropy uses Grimme's quasi-RRHO interpolation (Chem. Eur. J. 2012, 18,
9955: damping `w = 1 / (1 + (omega0 / omega)^alpha)` between the harmonic oscillator and a free rotor with
moment of inertia `mu' = mu Bav / (mu + Bav)`), and the vibrational enthalpy uses the Head-Gordon damping of the
ZPE-inclusive vibrational energy towards ½RT (Li et al., J. Phys. Chem. C 2015, 119, 1840), as in GoodVibes.
`bav: 1e-44` is Grimme's fixed value; `bav: auto` uses the molecule's own isotropically averaged moment of
inertia, which is what ORCA does. Printed `ZPE` and `E_0K` are always RRHO values.

The implementation was checked against ORCA 6.0.1 (`QuasiRRHO true`, three molecules) and GoodVibes 4.4
fed with the same frequencies: every RRHO term, the qRRHO entropy with `Bav: auto`, and the full qRRHO Gibbs
energy against GoodVibes agree to better than 3e-7 Eh. The only systematic difference to ORCA's Gibbs energy is
the Head-Gordon enthalpy damping, which ORCA does not apply (≈0.1 kcal/mol for a water dimer).

## Examples

```
examples/
  1-single_point/   1-basic, 2-dft-3c, 3-df, 4-pcm, 5-smd, 6-soscf (pyscf); 7-tblite, 8-xtb-solv, 9-anneal
  2-geometry/       1-optimization, 2-frequency, 3-constraints (pyscf)
  3-transition_state/   1-optts, 2-irc (tblite, Diels-Alder TS)
  4-nudged_elastic_band/   1-neb, 2-ci-neb, 3-autoneb (UMA)
  5-growing_string/ 1-gsm (tblite), 2-pyscf
  6-dynamics/       1-nve, 2-langevin, 3-nvt, 4-npt (tblite)
  7-pysisyphus/     1-optts, 2-irc (tblite through the bridge)
```

Run each from its directory with the command given at the bottom of the YAML (`qc_launch qc`, `md`, `neb`,
`autoneb`, `gsm` or `pysis`). The MACE / UMA examples point at local model files; adjust `model_path`.

## Development

```bash
python -m pytest -q                        # unit tests plus the integration tests whose backend is installed
python -m pytest -q -m "not slow and not gpu"
ruff check src tests && ruff format --check src tests
```

Integration tests are marked `pyscf`, `gpu`, `tblite`, `mace`, `uma`, `sella`, `pygsm`, `pysisyphus` and
`slow`, and skip automatically when the package (or the model file, via `QC_LAUNCHER_MACE_MODEL` /
`QC_LAUNCHER_UMA_MODEL`) is missing. Run the suite once per environment that provides a backend.
