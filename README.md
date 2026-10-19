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
| `qc` | one (last frame of `inputfile`) | optional `opt`, then a single point, optional `forces: true`, `freq`, `refine`, `irc`, in that order |
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

**GPU acceleration with NVIDIA cuEST** (pyscf driver, `accelerator: cuest`)

[cuEST](https://docs.nvidia.com/cuda/cuest) is NVIDIA's CUDA-X library for Gaussian-basis electronic structure
(beta, proprietary EULA). gpu4pyscf ships the integration (`gpu4pyscf.lib.cuest_wrapper`, gpu4pyscf >= 1.7); the
driver applies it on top of the density-fitted GPU method. Install the bindings matching your CUDA flavour:
`pip install -e .[cuest-cu13]` (CUDA 13; needed for Blackwell such as the GB10 and for cuEST's JIT path) or
`pip install -e .[cuest-cu12]` (CUDA 12, e.g. VANDA's A40 nodes with `module load CUDA/12.8.0`); cupy, gpu4pyscf
and cuEST must all be the same CUDA flavour, and `libnvrtc` / `libnvJitLink` must be discoverable by the loader.
cuEST keeps a JIT cache in `~/.cuest_cache/`.

```yaml
driver:
  name: pyscf
  xc: PBE0
  basis: def2-SVP
  with_df: true            # required
  with_gpu: true           # required
  accelerator: cuest
  cuest:                   # optional, defaults shown
    math_mode: native_fp64 # exact FP64; "default" = NVIDIA emulated / mixed precision (faster on A40, GB10)
    threshold_pq: 1.0e-14
    density_fitting_cutoff: 1.0e-12
    max_workspace_gb: 10
    disable: []            # any of hcore, jk, xc, nlc, pcm stays on native gpu4pyscf
    match_native_grids: false  # Stratmann partitioning, no radii adjustment, ISWIG PCM: for parity checks
```

What cuEST accelerates: core Hamiltonian (incl. ECP), density-fitted J/K (incl. range-separated hybrids),
XC and VV10 quadrature, C-PCM/COSMO, and all their nuclear gradients, so single points, optimizations, IRC,
NEB, GSM and MD. Limits enforced by the driver or the wrapper: density fitting and the GPU are required;
only C-PCM / COSMO solvation (no IEF-PCM, SS(V)PE, SMD); no `retry_soscf`; no composite `3c` methods; a fixed
functional list (HF, SVWN, BLYP, PBE, PBE0, B3LYP, B97, M06-L, M06, M06-2X, r2SCAN, B97M-V, wB97X, wB97X-V,
wB97M-V, CAM-B3LYP, HSE06, LC-wPBE, LC-wPBEh; the range-separated ones need gpu4pyscf newer than 1.8.1,
the 1.8.1 wheel's wrapper stops at r2SCAN / B97M-V); **no analytic Hessian**, so `freq` uses central differences of
the cuEST gradients by default and `numerical_hessian: false` computes a native gpu4pyscf Hessian at the same
geometry instead. Range-separated functionals agree with native gpu4pyscf only to ~1e-5 Eh (different DF
treatment); everything else matches to ~1e-10 Eh with `match_native_grids: true`. cuEST gradients always
include the grid-weight response, which native gpu4pyscf gradients omit by default; the difference is ~1e-4 eV/Å
on water and shrinks with finer grids. The cuEST EULA does not allow
publishing benchmark results without NVIDIA's permission, so this README gives none; `scripts/bench_accelerator.py`
measures native vs cuEST timings on your own hardware.

JoltQC (ByteDance-Seed) was evaluated as well and not integrated: the project is archived (June 2026), not on
PyPI, pinned to gpu4pyscf 1.4/1.5 internals, and accelerates only direct-SCF RHF/RKS energies (no density
fitting, gradients, Hessians, UKS or solvation), which none of the qc_launcher workflows benefit from.

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
(Å, pyscf symmetry detection tolerance), `qrrho` (true), `alpha` (4), `omega0` (100 cm⁻¹), `Bav` (1e-44 kg m²,
or `auto`). Writes `<stem>_vib.txt` (PyVibMS format) and `<stem>_thermo.json`. See *Thermochemistry* below.

**refine** (MP2 / CCSD / CCSD(T) single point on the final geometry, run in a separate interpreter):
`backend` (`pyscf` | `byteqc`), `python` (interpreter of the backend environment; default: the current one),
`pythonpath` (prepended for the runner, e.g. the parent directory of a ByteQC clone), `method` (`ccsd(t)` |
`ccsd` | `mp2`), `basis` (cc-pVTZ), `auxbasis`, `ecp`, `frozen_core` (true, or a number of frozen orbitals),
`conv_tol` (1e-8, CC amplitudes), `max_cycle` (100), `scf` (`{conv_tol: 1e-10, max_cycle: 100, density_fit:
true, auxbasis, gpu: false}`), `ngpu` (1), `gpu_memory_gb` / `cpu_memory_gb` (ByteQC CCSD buffers), `scratch_dir`,
`max_memory_mb`, `verbose` (3), `timeout_s`, `outputfile`. Closed-shell systems only. See *Energy refinement*.

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
| freq | `bav` | `Bav` |
| irc | `irc_trajectory`, `irc_output`, `steps`, `direction: backward` | `trajectory`, `outputfile`, `max_steps`, `direction: reverse` |
| md | `dtamp`; flat `optimizer`, `fmax`, `max_min_steps`, `min_trajectory`, `opt_kwargs` | `tdamp`; `minimize: {optimizer, fmax, max_steps, trajectory, kwargs}` |
| neb / autoneb | `max_neb_steps` / `steps`, `max_opt_steps` | `max_steps`, `endpoint_max_steps` |
| gsm | `num_nodes` / `max_nodes`, `max_gsm_steps` / `max_cycles`, `max_opt_steps`, `ID`, `fmax`, `ediff`, `line_search`, `endpoint_opt_steps`, `climb` | `nnodes`, `max_iters`, `opt_steps`, `string_id`, `conv_gmax`, `conv_ediff`, `linesearch`, `endpoint_max_steps`, `ts_search` |

## Outputs

| Task | Files (`<stem>` is the input file name without extension) |
|---|---|
| opt | `<stem>_opt.traj`, `<stem>_opt.xyz` |
| freq | `<stem>_vib.txt`, `<stem>_thermo.json`, optional `<stem>_hess.txt` (eV/Å², ASE layout) |
| refine | `<stem>_refine.json`, `<stem>_refine.log`, `<stem>_refine_input.json` / `<stem>_refine_raw.json` (runner exchange files) |
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
`Bav: 1e-44` is Grimme's fixed value; `Bav: auto` uses the molecule's own isotropically averaged moment of
inertia, which is what ORCA does. Printed `ZPE` and `E_0K` are always RRHO values.

The implementation was checked against ORCA 6.0.1 (`QuasiRRHO true`, three molecules) and GoodVibes 4.4
fed with the same frequencies: every RRHO term, the qRRHO entropy with `Bav: auto`, and the full qRRHO Gibbs
energy against GoodVibes agree to better than 3e-7 Eh. The only systematic difference to ORCA's Gibbs energy is
the Head-Gordon enthalpy damping, which ORCA does not apply (≈0.1 kcal/mol for a water dimer).

## Energy refinement

`refine` adds a correlated single-point energy on the geometry the driver produced, so a cheap driver (xTB, a
MLIP, DFT) supplies the structure and the thermal corrections and a high-level method supplies the electronic
energy. The calculation is RHF followed by DF-MP2, CCSD or CCSD(T) and runs out of process through
`src/qc_launcher/backends/refine_runner.py`, a self-contained script executed by the interpreter given as
`python`. Two backends share the runner:

- `pyscf`: CPU pyscf (`pyscf.mp.dfmp2_native`, `pyscf.cc`); works in the qc_launcher environment itself.
- `byteqc`: [ByteQC](https://github.com/bytedance/byteqc) (GPU DF-CCSD, CCSD(T) and DF-MP2; `cucc`,
  `cump2`). ByteQC pins `pyscf == 2.5.0`, which is older than what qc_launcher needs, so it lives in its own
  environment: `python` points at that environment's interpreter and `pythonpath` at the parent directory of
  the ByteQC clone (ByteQC is source-only). The runner's own imports are limited to the standard library,
  numpy and pyscf. ByteQC's DF-MP2 has no frozen-core option (a warning is printed); CCSD/CCSD(T) use `frozen_core`.
  By default the reference RHF runs on the CPU with pyscf (density fitted when `scf.density_fit` is true).
  `scf: {gpu: true}` runs it with ByteQC's `cuobc` RHF instead: exact integrals on the GPU, as in ByteQC's own
  examples, with the DF wrapper added afterwards for the correlated step.

When `freq` ran in the same `qc` call (or `<stem>_thermo.json` exists next to the input), the thermochemical
totals are recombined with the refined energy:

```
X_refined = X − E0 + E_refined        for X in E_0K, H_tot, G_tot, H_tot_qrrho, G_tot_qrrho
```

where `E0` is the driver's electronic energy, i.e. the driver's ZPE and thermal corrections are kept and only
the electronic energy is replaced. `<stem>_refine.json` holds `e_hf`, the correlation pieces (`e_corr_mp2`,
`e_corr_ccsd`, `e_t`), `e_total` (Eh), timings, the backend versions and the `composite` block.

A ByteQC environment that worked on an A40 (CUDA 12.8, x86_64):

```bash
conda create -n byteqc -c conda-forge python=3.11 "numpy<2" scipy h5py cmake make "cutensor>=2.1" cuda-version=12.8
conda activate byteqc
pip install pyscf==2.5.0 "cupy-cuda12x>=14" "nvmath-python[cu12]"
git clone https://github.com/bytedance/byteqc.git ~/repo/byteqc && cd ~/repo/byteqc
# ByteQC's CMake wants $CUTENSOR_ROOT/lib/12/libcutensor.so and $CUTENSOR_ROOT/include/cutensor.h
mkdir -p $CONDA_PREFIX/cutensor_root/lib && ln -sfn $CONDA_PREFIX/lib $CONDA_PREFIX/cutensor_root/lib/12 \
  && ln -sfn $CONDA_PREFIX/include $CONDA_PREFIX/cutensor_root/include
export CUTENSOR_ROOT=$CONDA_PREFIX/cutensor_root CUDA_ARCHITECTURES=86   # sm_86 = A40
for sub in cuobc cupbc; do cmake $sub/lib -B $sub/lib/build && make -j16 -C $sub/lib/build; done
echo ~/repo > $CONDA_PREFIX/lib/python3.11/site-packages/byteqc.pth   # or pass refine.pythonpath instead
python -c "import byteqc.cucc, byteqc.cump2"
```

On an aarch64 host with a CUDA 13 driver (DGX Spark / GB10, sm_121) the same recipe works with a conda CUDA 12.9
toolchain, except that the aarch64 CuPy wheel lacks the cuTENSOR bindings ByteQC imports, so CuPy is built
from source (about 15 minutes):

```bash
conda create -n byteqc-gb10 -c conda-forge python=3.11 scipy h5py cmake make "cutensor>=2.1" cuda-version=12.9 \
  cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-nvrtc-dev=12.9 cuda-profiler-api=12.9 libcublas-dev libcusolver-dev \
  libcusparse-dev libcurand-dev libcufft-dev "gcc_linux-aarch64=13" "gxx_linux-aarch64=13"
conda activate byteqc-gb10 && pip install pyscf==2.5.0 "nvmath-python[cu12]"
export CUDA_PATH=$CONDA_PREFIX CUTENSOR_PATH=$CONDA_PREFIX CUPY_NUM_BUILD_JOBS=16 \
  CUPY_NVCC_GENERATE_CODE="arch=compute_121,code=sm_121" CPATH=$CONDA_PREFIX/targets/sbsa-linux/include:$CONDA_PREFIX/include
pip install --no-binary cupy "cupy>=14"
# then build ByteQC as above with CUDA_ARCHITECTURES=121 and -DCMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc
```

Then in the YAML: `refine: {backend: byteqc, python: <env>/bin/python, method: ccsd(t), basis: cc-pVTZ}`.
The `byteqc`-marked tests run when `QC_LAUNCHER_BYTEQC_PYTHON` names that interpreter.

## Examples

```
examples/
  1-single_point/   1-basic, 2-dft-3c, 3-df, 4-pcm, 5-smd, 6-soscf, 10-cuest (pyscf); 7-tblite, 8-xtb-solv, 9-anneal
  2-geometry/       1-optimization, 2-frequency, 3-constraints (pyscf)
  3-transition_state/   1-optts, 2-irc (tblite, Diels-Alder TS)
  4-nudged_elastic_band/   1-neb, 2-ci-neb, 3-autoneb (UMA)
  5-growing_string/ 1-gsm (tblite), 2-pyscf
  6-dynamics/       1-nve, 2-langevin, 3-nvt, 4-npt (tblite)
  7-pysisyphus/     1-optts, 2-irc (tblite through the bridge)
  8-refinement/     1-xtb_opt_ccsd_t (tblite opt + freq, ByteQC CCSD(T)), 2-pyscf_backend (CPU pyscf)
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
