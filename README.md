# Bayesian Orbit Determination via Markov Chain Monte Carlo with High-Order Flow Expansions

**Giovanni Fereoli<sup>1,2</sup> · Jay W. McMahon<sup>1,2</sup>**  
<sup>1</sup>Ann and H.J. Smead Department of Aerospace Engineering Sciences, University of Colorado Boulder  
<sup>2</sup>Colorado Center for Astrodynamics Research (CCAR), University of Colorado Boulder  

---

## Overview

This project introduces a **Bayesian orbit determination framework** for highly nonlinear and weakly observable small-body environments, accelerated using **high-order State Transition Tensors (STTs)**.

The methodology enables efficient posterior sampling by replacing repeated nonlinear propagation with a locally valid algebraic deviation map about a reference trajectory. The framework integrates:

- **Markov Chain Monte Carlo (MCMC)** posterior sampling  
- **STT-based fast likelihood evaluation**  
- **Batch least-squares warm starts** (square-root information filter)  
- **State-space whitening for correlated parameters**  
- **Standard convergence diagnostics**

The framework is demonstrated on **radio-only and angles-only orbit determination case studies** in the vicinity of asteroid 101955 Bennu: a radio probe on a low orbit (Case 1) and a particle ejected from the surface (Cases 2 and 3).


---

## Paper cases

Each paper case has one script at the repository root and one results folder in `results/`. The folder name starts with the case label, and the timestamp is the run time (`YYMMDDHHMMSS`).

| Case | Script | Results | Setup |
| --- | --- | --- | --- |
| 1a | `case1_bennu_popper.py` (`CASE = "1a"`) | `results/case1a_bennu_popper_261126094521/` | Radio probe released by a spring onto a low polar orbit about Bennu, one-way Doppler from OSIRIS-REx (24 h), Gaussian priors on the degree-2 coefficients. The C22–S22 posterior is non-Gaussian (a crescent). Adds the GMM + STM propagation to 48 h |
| 1b | `case1_bennu_popper.py` (`CASE = "1b"`) | `results/case1b_bennu_popper_uniform_261126094521/` | Case 1a with uniform (box) priors on C20, C21, S21, C22, S22, without the GMM propagation |
| 2a | `case2_bennu_particle.py 2a` | `results/case2a_bennu_particle_stt2_261126112204/` | Particle launched from near Bennu's equator (lat 3°), OSIRIS-REx optical angles (RA/Dec, 22 images over 1.5 h) taken from below the south pole, along the spin axis. The batch (SRIF, full rank) converges to a shallow local minimum 26 m from the truth (cost 2.5 above the best fit). Second-order STT likelihood: the posterior has two lobes, one at the batch and one at the truth |
| 2b | `case2_bennu_particle.py 2b` | `results/case2b_bennu_particle_stmbank_261126101730/` | Case 2a with first-order STMs only: a bank of 21 STMs along the batch's least-determined position direction. It also shows both lobes |
| 2 (comparison) | `case2_comparison.py` | `results/case2_comparison_261126112203/` | The Case 2 posterior under four likelihoods: exact dynamics (sampled by this script), STT-2 (2a), STM bank (2b) and a single STM about the batch (`results/case2_bennu_particle_stm_261126101730/`, from `case2_bennu_particle.py stm`). Only the exact, STT-2 and bank likelihoods find the truth lobe |
| 2 (seed MC) | `case2_seed_mc.py` | `results/case2_seed_mc_261127000315/` | The Case 2b batch over 2000 other noise and reference seed pairs: how often the cost along the launch height has two minima, and how often the batch stops in the wrong one (see [Seed Monte Carlo](#seed-monte-carlo-case-2)) |
| 3 | `case3_bennu_particle_hierSH.py` | `results/case3_bennu_particle_hierSH_261126152614/` | Case 2a with hierarchical priors: the prior scale of each SH order (m = 0, 1, 2) is sampled with the other parameters, under a log-normal hyperprior |
| Extra | `extra_ryugu_probe.py` | `results/extra_ryugu_probe_261122091827/` | Ryugu-shaped body (polyhedron, scaled to 289P/Blanpain size, Sun + SRP), ejected particle observed by a nearby spacecraft (angles), GMM + STM propagation |
| STT benchmark | `stt_benchmark.py` | `results/stt_benchmark_261119213307/` | STT orders 1–3 against the full nonlinear likelihood on the samples of an MCMC chain (two-body, angles) |

Run the scripts from the repository root:

```bash
python case1_bennu_popper.py                 # Case 1a or 1b, set CASE in the settings
python case2_bennu_particle.py 2a            # Case 2a (STT-2 likelihood)
python case2_bennu_particle.py 2b            # Case 2b (bank of STMs)
python case2_bennu_particle.py stm           # single STM, for the comparison
python case2_comparison.py                   # samples the exact reference (4 seeds in parallel, up to 10^6 steps each, ~9 h), compares
python case2_comparison.py --exact results/case2_comparison_<stamp>/exact_reference.npz
python case3_bennu_particle_hierSH.py        # Case 3
python case2_seed_mc.py                      # seed Monte Carlo of the Case 2 batch (2000 realizations, ~25 min on 8 processes)
```

`case2_comparison.py` takes the newest 2a, 2b and stm folders, or the ones given with `--runs DIR_2A DIR_2B DIR_STM`. With `--exact` it reuses an exact reference and writes the figure and the table next to it. `case2_bennu_particle.py <case> --post <folder>` redoes only the post-processing of a finished run (the clean RTN corner and the summary).

To redraw the figures from a saved chain without sampling again, set `LOAD_CHAIN` in the script to that folder's `chain.npz`. For Case 1, also set `CASE` to the case of that run. For the STT benchmark, `LOAD_BENCHMARK` does the same for `benchmark.npz`.

All the paper cases share the integration and MCMC settings: DOP853 with `rtol = 1e-11`, `atol = 1e-14` for every propagation; 128 walkers started from N(batch, batch covariance) (spread 1 in the batch-whitened coordinates, seeded with `SEED_WALKERS = 7`); at least 100000 steps, then on in blocks of 50000 until the run spans 50 integrated autocorrelation times τ (at most 10<sup>6</sup> steps); burn-in 2 τ<sub>max</sub> and thinning 0.5 τ<sub>min</sub>. Walkers whose median log-posterior is more than 100 below the ensemble median (stranded) are left out of τ and of the statistics.

Bennu's rotation is the IAU model of the SPICE kernel `bennu_v17.tpc` (frame `IAU_BENNU`: pole RA 85.45864°, Dec −60.36468°, period 4.296004 h, prime meridian W(t) from the kernel). The body-fixed launch sites and the degree-2 field are in that frame.

The MCMC runs on all the CPUs. To share the machine between runs, set the environment variable `MCMC_PROCS` to the number of processes of each run (the chain does not depend on it). Run unattended jobs with `MPLBACKEND=Agg`, or `plt.show()` blocks. Each `run.log` has `[Timing]` lines (batch, STT/STM propagation, MCMC sampling with ms per step and per evaluation, total), and `chain.npz` stores `run_wall_s` and `n_procs`.

`case1_bennu_popper.py` imports its degree-2 dynamics and helpers from `extra/scenario_ISL_SH2.py`. The SPICE kernels in `data/kernels/` are not tracked.

### Valley coordinates (`--valley`)

The Case 2 and Case 3 posteriors have two lobes along a bent valley in launch height. The stretch move mixes slowly between them: τ grows with the run, and the runs above stop at 10<sup>6</sup> steps with n/τ = 11–14.

The `--valley` option samples the same posterior in valley-straightening coordinates. It is available in `case2_bennu_particle.py`, `case3_bennu_particle_hierSH.py` and `case2_comparison.py`, and is off by default. The coordinates come from a triangular transport map of the batch-whitened coordinates z:

z = h e + Q (y\*(h) + C(h) w)

- h is the coordinate along the batch's least-determined position direction (the top eigenvector of the position block of the batch covariance).
- y\*(h) is the profile path: the best other coordinates at fixed h, from Levenberg–Marquardt on the likelihood and prior residuals, with step 0.1, until the profile falls 25 below its best.
- C(h) is the Cholesky factor of the Gauss–Newton conditional covariance there.

The log-posterior gets the Jacobian term log |dz/dv| = Σ log diag C(h). The walkers move in v = (h, w), with the stretch move and the shared settings above. The chain is mapped back to z before τ, burn-in, thinning and saving, so the outputs have the same form as without the option. In the library, `MCMCModel.setup_valley_map` builds the map, using `MCMC.ValleyMap` and `MCMC.build_valley_map`. `chain.npz` stores the map.

```bash
python case2_bennu_particle.py 2a --valley
python case2_bennu_particle.py 2b --valley
python case2_comparison.py --valley --exact-only       # exact reference in valley coordinates
python case2_comparison.py --valley --exact results/case2_comparison_<stamp>_valley/exact_reference.npz --runs DIR_2A_VALLEY DIR_2B_VALLEY DIR_STM
python case3_bennu_particle_hierSH.py --valley
```

Results (the single STM posterior is Gaussian and keeps its plain run). The Case 2 valley runs stopped at the 100000-step minimum; Case 3 mixes more slowly (τ 1800–4700 in every coordinate) and needed 250000 steps; the map straightens only the launch-height valley, and the three log σ<sub>g</sub> coordinates enter it through the Gaussian conditional C(h) only. The MCMC time is the wall time with other runs sharing the machine.

| Case | Results | τ<sub>max</sub> [steps] | n/τ | MCMC time (stretch run, 10<sup>6</sup> steps) |
| --- | --- | --- | --- | --- |
| 2a | `results/case2a_bennu_particle_stt2_261126201332_valley/` | 299 | 335 | 0.43 h on 3 processes (7.13 h) |
| 2b | `results/case2b_bennu_particle_stmbank_261126193440_valley/` | 255 | 392 | 0.55 h on 3 processes (4.66 h on 5) |
| 2 (comparison) | `results/case2_comparison_261126200658_valley/` | 329–341 (exact, 4 seeds) | 294 | 0.89 h per seed on one process (8.69 h) |
| 3 | `results/case3_bennu_particle_hierSH_261126204031_valley/` | 4747 | 53 | 0.56 h on 5 processes, 250000 steps (3.73 h on 5) |

The two samplers agree on the posterior shape: the corner plots of the 2b runs match. The valley runs give the converged lobe shares:

| Likelihood | P(truth lobe), stretch, 10<sup>6</sup> steps | P(truth lobe), valley |
| --- | --- | --- |
| Exact dynamics | 0.694 (seeds 0.669–0.714) | 0.741 (seeds 0.739–0.743) |
| STT-2 (2a) | 0.517 | 0.592 |
| Bank of 21 STMs (2b) | 0.718 | 0.747 |
| Single STM | 0.001 | 0.001 |

For Case 3 the per-group SH scales σ<sub>g</sub> agree between the two samplers to within 2 % (zonal 4.32e-4 vs 4.38e-4, tesseral 1.99e-16 vs 2.01e-16, sectoral 1.50e-5 vs 1.49e-5; posterior medians).

### Seed Monte Carlo (Case 2)

Case 2 uses one noise realization (noise seed 395, reference seed 239), chosen so that the batch stops in the minimum farther from the truth. `case2_seed_mc.py` puts a rate on that outcome. It repeats the Case 2b batch over 2000 fresh seed pairs (noise 100000–101999, reference 200000–201999). The geometry, priors and batch settings are those of the 2b run (SRIF, full rank, step halving, `max_iter` 30). For each realization it does three things:

- **Batch.** `BatchLSQ.gn_with_stm` runs as is. Only its propagator changes: the exact degree-2 dynamics with a central-difference STM. This reproduces the saved 2b batch exactly (`--validate`) in 0.1 s.
- **Continuation.** The same Gauss–Newton runs 300 more iterations from the batch result. This separates a converged batch from one that stopped at 30 iterations.
- **Profile.** It profiles the cost J = −2 ln p along the launch height h<sub>0</sub> (−60 to 80 m, step 1 m), minimizing over the other 11 coordinates, then refines each local minimum.

The run took 23.2 min on 8 processes, a median of 5.3 s per realization on one core. The table gives 95 % binomial intervals. A second minimum counts when its barrier is at least 0.5 in J.

| Result | Rate |
| --- | --- |
| Profile with two or more minima | 7.1 % ± 1.1 (13.5 % counting barriers < 0.5) |
| Batch converged at the best fit | 87.8 % ± 1.4 |
| Batch converged at another minimum | 5.1 % ± 1.0 (farther from the truth than the best fit, as in Case 2: 1.8 % ± 0.6) |
| Batch stopped short at 30 iterations, still descending | 4.6 % ± 0.9 |
| Reference trajectory falls back to Bennu (batch fails) | 2.5 % ± 0.7 |
| Batch at the wrong minimum, given two minima | 40.1 % ± 8.1 |
| As in Case 2: farther minimum, 2–3 above the best fit, ≥ 20 m away | 0.2 % ± 0.2 (5 of 2000) |

The Case 2 realization is rare on both counts:

- **Its reference.** It sits +80.4 m along the pole (3.0 σ). None of the 2000 Monte Carlo references is that far.
- **Its outcome.** The rate of batches stopping at the farther minimum grows with the reference offset, from 0 % below −50 m to 6.4 % above +50 m.

`summary.txt` has the full table, including the rates by reference offset. To redo the summary and the figure from a finished run, without running the batches again, use:

```bash
python case2_seed_mc.py --load results/case2_seed_mc_<stamp>/mc.npy
```

### What is in each results folder

Every MCMC folder has `chain.npz` (the MCMC chain), and every folder has one figure per `fig_*.pdf` file. Most also have `run.log`, the screen output of the run.

| File | Content | Cases |
| --- | --- | --- |
| `fig_scene_orbit.pdf` | The probe's orbit about Bennu and the spacecraft | 1a, 1b |
| `fig_scene_body_fixed.pdf` | The particle's trajectory in the body-fixed frame, on the shape model | 2a, 2b, stm, 3, Extra |
| `fig_observation_geometry.pdf` | Measurement geometry over the arc | all but the benchmark |
| `fig_postfit_residuals.pdf` | Post-fit residuals of the batch and of the MCMC | all but the benchmark |
| `fig_mcmc_traces.pdf`, `fig_mcmc_autocorrelation.pdf` | Walker traces and autocorrelation | all |
| `fig_corner.pdf`, `fig_corner_prior.pdf` | Corner plot of the estimated parameters, without and with the prior | all |
| `fig_corner_rtn.pdf` | Corner plot with the position and velocity deviations in the RTN frame | all but 2a, 2b, stm |
| `fig_corner_rtn_<tag>.pdf` | The RTN corner plot without the stranded walkers (`tag` = `stt2`, `stmbank`, `stm`) | 2a, 2b, stm |
| `fig_corner_sh2.pdf`, `fig_corner_sh2_prior.pdf` | Corner plot of the degree-2 coefficients (the C22–S22 crescent) | 1a, 1b |
| `fig_corner_derived.pdf`, `fig_corner_derived_prior.pdf` | Corner plot of derived quantities | 1a, 1b |
| `fig_gmm_stm_propagation.pdf`, `fig_gmm_stm_propagation_rtn.pdf`, `fig_gmm_prediction_history.pdf` | GMM split of the posterior and its STM propagation past the arc | 1a, Extra |
| `batch.npz` | Batch solution, data and dynamics constants, read by `case2_comparison.py` | 2a, 2b, stm |
| `fig_case2_comparison.pdf`, `summary.txt`, `exact_reference.npz` | The four Case 2 posteriors in launch height, out-of-plane speed and GM; the table of lobe shares and truth credible levels; the exact-dynamics chains | 2 (comparison) |
| `fig_case2_seed_mc.pdf`, `summary.txt`, `mc.npy` | Where the batch stops relative to the best fit, by outcome, against the reference offset; the cost gap of the second minimum; the table of rates; one record per realization (seeds, costs, profile, minima) | 2 (seed MC) |
| `fig_stt_benchmark.pdf`, `benchmark.npz` | STT accuracy and cost against the full likelihood | STT benchmark |

### Other runs

The other folders in `results/` are runs that are not in the paper:

- `extra_ryugu_probe_*_priorx100/`: the extra case with position and velocity prior sigmas 100 times larger (`PRIOR_SCALE = 100`).
- `get_interior_*/`: interior density and inertia fits (`get_interior_*.py`).

The runs of the scripts in `extra/` are in `extra/results/`.

### Library

- `MCMC.py`: `MCMCModel`; `ValleyMap` and `build_valley_map` (valley-straightening coordinates, `--valley`)
- `STTPropagationND.py`: STT propagation
- `BatchLSQ.py`: batch least squares as a square-root information filter (SRIF, Householder; `solver="ud"`: Bierman U-D), Gauss-Newton iterations on the exact MAP cost (no damping, no scaling), full rank, with step halving (all the paper cases)
- `GMM.py`: GMM split and propagation
- `MCMCPlots.py`: journal figures

---

## Citation

If you use this code or reproduce results from the paper, please cite:

```bibtex
@article{Fereoli2026_MCMCSTTOD,
  title   = {Bayesian Orbit Determination via Markov Chain Monte Carlo with High-Order Flow Expansions},
  author  = {Fereoli, Giovanni and McMahon, Jay W.},
  journal = {Acta Astronautica},
  year    = {2026},
  note    = {In Preparation}
}
```

---

## License

Released under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
