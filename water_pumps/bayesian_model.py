import jax.numpy as jnp
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist


def uninformed_model(X: npt.NDArray, regions: npt.NDArray, y: npt.NDArray):
    n, p = X.size()
    n_regions = len(regions)

    # Uninformed hyper-priors for the one-level hierarchical model
    μ_a = numpyro.sample('μ_a', dist.Normal(0., 100.))
    σ_a = numpyro.sample('σ_a', dist.HalfCauchy(5.))

    μ_t = numpyro.sample('μ_t', dist.Normal(0., 100.))
    σ_t = numpyro.sample('σ_t', dist.HalfCauchy(5.))
    μ = jnp.repeat([μ_t], repeats=p)
    Σ = jnp.diag(jnp.repeat([σ_t], repeats=p))

    # Create an intercept for each administrative region
    with numpyro.plate('regions', n_regions):
        α = numpyro.sample('α', dist.Normal(μ_a, σ_a))
        θ = numpyro.sample('θ', dist.MultivariateNormal(μ, Σ))
    
    β = α[regions] + jnp.dot(X, θ[regions])
    with numpyro.plate('samples', n):
        numpyro.sample('y', dist.Bernoulli(logits=β), obs=y)
