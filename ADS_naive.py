# NOTE: more on this at https://github.com/giovannipurpura/daceypy/blob/master/daceypy/_ADS.py


def get_admissible_bounds(param_priors):
    """
    Returns (lower, upper) bounds for each parameter from the prior CDF.
    """
    bounds = []
    for p in param_priors:
        lower = p.ppf(1e-6)
        upper = p.ppf(1 - 1e-6)
        bounds.append((lower, upper))
    return np.array(bounds)  # shape (ndim, 2)


def split_domain(bounds, n_splits):
    """
    Split each dimension of the domain into equal intervals.
    Returns a list of subdomain centers.
    """
    ndim = bounds.shape[0]
    grid_axes = [
        np.linspace(bounds[i, 0], bounds[i, 1], n_splits + 1) for i in range(ndim)
    ]

    # Compute midpoints for each cell
    cell_centers = []
    for idx in np.ndindex(*([n_splits] * ndim)):
        center = [
            0.5 * (grid_axes[d][idx[d]] + grid_axes[d][idx[d] + 1]) for d in range(ndim)
        ]
        cell_centers.append(np.array(center))
    return cell_centers


def run_mcmc_per_subdomain(
    model_template, cell_centers, spherical_spread=1e-4, n_samples=5000, n_walkers=40
):
    """
    Launch one MCMC chain from each admissible region subdomain.
    """
    all_models = []

    for i, center in enumerate(cell_centers):
        print(f"\n=== Chain {i+1}/{len(cell_centers)}: starting from {center} ===")

        mcmc = MCMCModel(
            residuals_func=model_template.residuals_func,
            initial_params=center,
            param_priors=model_template.param_priors,
            observed_data=model_template.observed_data,
        )

        if getattr(model_template, "is_whitened", False):
            mcmc.setup_whitening(
                cov=model_template.whiten_L @ model_template.whiten_L.T
            )

        mcmc.run(
            n_samples=n_samples,
            n_walkers=n_walkers,
            spherical_spread=spherical_spread,
        )
        all_models.append(mcmc)

    return all_models
