from jax import random, vmap
from jax.example_libraries.optimizers import adam
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from scipy.stats import beta, norm
from sklearn.linear_model import LogisticRegression


def compute_logit(α, θ, x):
    return α + jnp.dot(θ, x)


def uninformed_model(X: npt.NDArray, regions: npt.NDArray, y: npt.NDArray):
    n, p = X.shape
    n_regions = len(np.unique(regions))

    # Uninformed hyper-priors for the one-level hierarchical model
    μ_a = numpyro.sample('μ_a', dist.Normal(0., 100.))
    σ_a = numpyro.sample('σ_a', dist.HalfCauchy(5.))

    μ_t = numpyro.sample('μ_t', dist.Normal(0., 100.))
    σ_t = numpyro.sample('σ_t', dist.HalfCauchy(5.))
    μ = jnp.repeat(μ_t, repeats=p)
    Σ = jnp.diag(jnp.repeat(σ_t, repeats=p))

    # Create an intercept for each administrative region
    with numpyro.plate('regions', n_regions):
        α = numpyro.sample('α', dist.Normal(μ_a, σ_a))
        θ = numpyro.sample('θ', dist.MultivariateNormal(μ, Σ))
    
    β = vmap(compute_logit, in_axes=0)(α[regions], θ[regions], X)
    with numpyro.plate('samples', n):
        numpyro.sample('y', dist.Bernoulli(logits=β), obs=y)


def compute_region_params(y: npt.NDArray):
    m = y.mean()
    α = len(y) * m
    β = len(y) - α

    samples = beta.rvs(α, β, size=1000, random_state=17)
    logit_samples = np.log(samples / (1 - samples))
    μ, σ = norm.fit(logit_samples)
    return μ, σ


def compute_feature_params(X: npt.NDArray, y: npt.NDArray):
    model = LogisticRegression(fit_intercept=False, random_state=17)
    model.fit(X, y)
    θ = model.coef_
    
    w = X @ θ.T
    w = jnp.exp(w) / jnp.square(1 + jnp.exp(w))
    jnp.diag(w).shape
    return θ, X.T @ jnp.diag(w.flatten()) @ X


def empirical_model(X: npt.NDArray, regions: npt.NDArray, y: npt.NDArray,
                    μ_a: float, σ_a: float, μ: npt.NDArray, τ: npt.NDArray):
    n, _ = X.shape
    n_regions = len(np.unique(regions))

    with numpyro.plate('regions', n_regions):
        α = numpyro.sample('α', dist.Normal(μ_a, σ_a))
        θ = numpyro.sample('θ', dist.MultivariateNormal(μ, precision_matrix=τ))

    β = vmap(compute_logit, in_axes=0)(α[regions], θ[regions], X)
    with numpyro.plate('samples', n):
        numpyro.sample('y', dist.Bernoulli(logits=β), obs=y)


def fit_model(X: npt.NDArray, regions: npt.NDArray, y:npt.NDArray, **kwargs):
    # Giving sensible defaults for MCMC arguments, but allowing user to pass
    # different ones if they want
    default_kwargs = {'seed': 17, 'prior': 'uninformed', 'step_size': 0.005,
                      'n_steps': 10000}
    kwargs = {**default_kwargs, **kwargs}

    rng_key = random.PRNGKey(kwargs['seed'])

    if kwargs['prior'] == 'uninformed':
        guide = AutoNormal(uninformed_model)
        model = SVI(uninformed_model, guide, 
                    adam(step_size=kwargs['step_size']), Trace_ELBO())
    else:
        μ_a, σ_a = compute_region_params(y)
        μ, τ = compute_feature_params(X, y)

        guide = AutoNormal(empirical_model)
        model = SVI(empirical_model, guide,
                    adam(step_size=kwargs['step_size']), Trace_ELBO())
    
    model_result = model.run(rng_key, kwargs['n_steps'], X, regions, y, 
                             μ_a, σ_a, μ, τ)

    return model_result, guide
