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
| 1a | `case1_bennu_popper.py` (`CASE = "1a"`) | `results/case1a_bennu_popper_261122133644/` | Radio probe released by a spring onto a low polar orbit about Bennu, one-way Doppler from OSIRIS-REx (24 h), Gaussian priors on the degree-2 coefficients. The C22–S22 posterior is non-Gaussian (a crescent). Adds the GMM + STM propagation to 48 h |
| 1b | `case1_bennu_popper.py` (`CASE = "1b"`) | `results/case1b_bennu_popper_uniform_261122133644/` | Case 1a with uniform (box) priors on C20, C21, S21, C22, S22, without the GMM propagation |
| 2a | `case2_bennu_particle.py 2a` | `results/case2a_bennu_particle_stt2_261125163543/` | Particle launched from Bennu's equator, OSIRIS-REx optical angles (RA/Dec, 22 images over 1.5 h) taken from below the south pole, along the spin axis. The batch (SRIF, full rank) converges to a shallow local minimum 18.5 m from the truth. Second-order STT likelihood: the posterior has two lobes, one at the batch and one at the truth |
| 2b | `case2_bennu_particle.py 2b` | `results/case2b_bennu_particle_stmbank_261125163725/` | Case 2a with first-order STMs only: a bank of 21 STMs along the batch's least-determined position direction. It also shows both lobes |
| 2 (comparison) | `case2_comparison.py` | `results/case2_comparison_261125164903/` | The Case 2 posterior under four likelihoods: exact dynamics (sampled by this script), STT-2 (2a), STM bank (2b) and a single STM about the batch (`results/case2_bennu_particle_stm_261125163728/`, from `case2_bennu_particle.py stm`). Only the exact, STT-2 and bank likelihoods find the truth lobe |
| 3 | `case3_bennu_particle_hierSH.py` | `results/case3_bennu_particle_hierSH_261125163543/` | Case 2a with hierarchical priors: the prior scale of each SH order (m = 0, 1, 2) is sampled with the other parameters, under a log-normal hyperprior |
| Extra | `extra_ryugu_probe.py` | `results/extra_ryugu_probe_261122091827/` | Ryugu-shaped body (polyhedron, scaled to 289P/Blanpain size, Sun + SRP), ejected particle observed by a nearby spacecraft (angles), GMM + STM propagation |
| STT benchmark | `stt_benchmark.py` | `results/stt_benchmark_261119213307/` | STT orders 1–3 against the full nonlinear likelihood on the samples of an MCMC chain (two-body, angles) |

Run the scripts from the repository root:

```bash
python case1_bennu_popper.py                 # Case 1a or 1b, set CASE in the settings
python case2_bennu_particle.py 2a            # Case 2a (STT-2 likelihood)
python case2_bennu_particle.py 2b            # Case 2b (bank of STMs)
python case2_bennu_particle.py stm           # single STM, for the comparison
python case2_comparison.py                   # samples the exact reference (4 seeds in parallel, ~1 h), compares
python case2_comparison.py --exact results/case2_comparison_<stamp>/exact_reference.npz
python case3_bennu_particle_hierSH.py        # Case 3
```

`case2_comparison.py` takes the newest 2a, 2b and stm folders, or the ones given with `--runs DIR_2A DIR_2B DIR_STM`. With `--exact` it reuses an exact reference and writes the figure and the table next to it. `case2_bennu_particle.py <case> --post <folder>` redoes only the post-processing of a finished run (the clean RTN corner and the summary).

To redraw the figures from a saved chain without sampling again, set `LOAD_CHAIN` in the script to that folder's `chain.npz`. For Case 1, also set `CASE` to the case of that run. For the STT benchmark, `LOAD_BENCHMARK` does the same for `benchmark.npz`.

`case1_bennu_popper.py` imports its degree-2 dynamics and helpers from `extra/scenario_ISL_SH2.py`. The SPICE kernels in `data/kernels/` are not tracked.

### What is in each results folder

Every folder has `chain.npz` (the MCMC chain) and one figure per `fig_*.pdf` file. Most also have `run.log`, the screen output of the run. The Case 3 folder also has `replot.log`, the screen output of the figures redrawn from its `chain.npz`.

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
| `fig_stt_benchmark.pdf`, `benchmark.npz` | STT accuracy and cost against the full likelihood | STT benchmark |

### Other runs

The other folders in `results/` are runs that are not in the paper:

- `extra_ryugu_probe_*_priorx100/`: the extra case with position and velocity prior sigmas 100 times larger (`PRIOR_SCALE = 100`).
- `get_interior_*/`: interior density and inertia fits (`get_interior_*.py`).

The runs of the scripts in `extra/` are in `extra/results/`.

### Library

- `MCMC.py`: `MCMCModel`
- `STTPropagationND.py`: STT propagation
- `BatchLSQ.py`: batch least squares as a square-root information filter (SRIF, Householder; `solver="ud"`: Bierman U-D), Gauss-Newton iterations on the exact MAP cost (no damping, no scaling), full rank, with step halving by default. Cases 2 and 3 take full steps (`halving=False`)
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
