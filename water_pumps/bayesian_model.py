from jax import random
import jax.numpy as jnp
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


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


def empirical_model(X: npt.NDArray, regions: npt.NDArray, y: npt.NDArray):
    return None


def fit_model(X: npt.NDArray, regions: npt.NDArray, y:npt.NDArray, **kwargs):
    # Giving sensible defaults for MCMC arguments, but allowing user to pass
    # different ones if they want
    default_kwargs = {'seed': 17, 'prior': 'uninformed', 'n_warmup': 1000,
                      'n_samples': 2000}
    kwargs = {**default_kwargs, **kwargs}

    rng_key = random.PRNGKey(kwargs['seed'])

    if kwargs['prior'] == 'uninformed':
        kernel = NUTS(uninformed_model)
    else:
        kernel = NUTS(empirical_model)

    model = MCMC(kernel, num_warmup=kwargs['n_warmup'], num_samples=kwargs['n_samples'])
    model.run(rng_key, X, regions, y)
    return model
