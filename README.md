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
- **Batch least-squares warm starts**  
- **State-space whitening for correlated parameters**  
- **Standard convergence diagnostics**

The approach is demonstrated on an **angles-only optical navigation case study** involving particle trajectory reconstruction near asteroid 101955 Bennu.

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
