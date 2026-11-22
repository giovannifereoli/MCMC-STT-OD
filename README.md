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

The framework is demonstrated on **angles-only and radio-only orbit determination case studies** involving particle trajectory reconstruction in the vicinity of asteroid 101955 Bennu.


---

## Paper cases

Each paper case has one script at the repository root and one results folder in `results/`. The folder name starts with the case label, and the timestamp is the run time (`YYMMDDHHMMSS`).

| Case | Script | Results | Setup |
| --- | --- | --- | --- |
| 1a | `case1a_bennu_particle.py` | `results/case1a_bennu_particle_261122095204/` | Particle ejected from Bennu, OSIRIS-REx optical angles (RA/Dec, 22 epochs over 1.5 h), first-order STT (STM) likelihood |
| 1b | `case1b_bennu_particle_stt2.py` | `results/case1b_bennu_particle_stt2_261122095204/` | Case 1a with a second-order STT likelihood |
| 2a | `case2_bennu_popper.py` (`CASE = "2a"`) | `results/case2a_bennu_popper_261122133644/` | Radio probe released by a spring onto a low polar orbit, one-way Doppler from OSIRIS-REx (24 h), Gaussian priors on the degree-2 coefficients. Shows a non-Gaussian C22–S22 posterior (crescent) and adds the GMM + STM propagation to 48 h |
| 2b | `case2_bennu_popper.py` (`CASE = "2b"`) | `results/case2b_bennu_popper_uniform_261122133644/` | Case 2a with uniform (box) priors on C20, C21, S21, C22, S22, without the GMM propagation |
| 3 | `case3_bennu_particle_hierSH.py` | `results/case3_bennu_particle_hierSH_261122095204/` | Case 1a with hierarchical priors: the prior scale of each SH order (m = 0, 1, 2) is learned from the data |
| Extra | `extra_ryugu_probe.py` | `results/extra_ryugu_probe_261122091827/` | Ryugu-shaped body (polyhedron, scaled to 289P/Blanpain size, Sun + SRP), ejected particle observed by a nearby spacecraft (angles), GMM + STM propagation |
| STT benchmark | `stt_benchmark.py` | `results/stt_benchmark_261119213307/` | STT orders 1–3 against the full nonlinear likelihood on the samples of an MCMC chain (two-body, angles) |

Each results folder contains:

- `run.log`: the screen output of the run (batch, MCMC and check summaries).
- `chain.npz`: the MCMC chain.
- `fig_*.pdf`: one figure per file.

To redraw the figures from the saved chain without sampling again, set `LOAD_CHAIN` in the script to that folder's `chain.npz`. For case 2, also set `CASE` to the case of that run. For the STT benchmark, `LOAD_BENCHMARK` does the same for `benchmark.npz`.

Run the scripts from the repository root. `case2_bennu_popper.py` imports its degree-2 dynamics and helpers from `extra/scenario_ISL_SH2.py`. The SPICE kernels in `data/kernels/` are not tracked.

The other folders in `results/` are runs that are not in the paper:

- `case2a_bennu_popper_*_circular/`, `case2b_bennu_popper_uniform_*_circular/`: case 2 without the release error (`DELIVERY = False`).
- `extra_ryugu_probe_*_priorx100/`: the extra case with position and velocity prior sigmas 100 times larger (`PRIOR_SCALE = 100`).
- `bennu_particle_nominal23_*/`: `case1b_bennu_particle_stt2.py nominal` (23-epoch arc, first order).
- `bennu_a2_popper_*/`: `J1_scenarioA2_doppler_SPH2_Bennu_popper.py`.
- `get_interior_*/`: interior density and inertia fits (`get_interior_*.py`).

The ISL runs of `extra/scenario_ISL_SH2.py` are in `extra/results/`. The shared library is:

- `MCMC.py`: `MCMCModel`
- `STTPropagationND.py`: STT propagation
- `BatchLSQ.py`: batch least squares as a square-root information filter (SRIF, Householder; `solver="ud"`: Bierman U-D), Gauss-Newton iterations with step halving on the exact MAP cost (no damping, no scaling). Cases 1a, 1b and 3 use `rank_cutoff=True, halving=False`: the singular-value cutoff of `numpy.linalg.lstsq` and full steps, which give the same batch as the earlier `lstsq` implementation
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
