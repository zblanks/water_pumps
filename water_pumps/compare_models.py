import arviz as az
import bayesian_model as bm
import dill
from jax import random
from numpyro.infer import log_likelihood
import os


def compute_logprob(model, guide, *args):
    post = guide.sample_posterior(random.PRNGKey(17), model.params, (1000,))
    logprob = log_likelihood(guide.model, post, *args)

    # ArViz requires the shape to be (n_chains, n_draws, *dims) so I'm going to
    # imply there is one chain for our posterior sampling
    for key in post.keys():
        post[key] = post[key].reshape(1, *post[key].shape)

    res = az.from_dict(
        posterior=post,
        log_likelihood={k: v[None] for k, v in logprob.items()}
    )
    return res


def compare_models(X, regions, y) -> dict:
    # To save computation in the future create a models folder where we store
    # the SVI models and their parameters
    if not os.path.isdir('models'):
        os.mkdir('models')

    model_dict = {'uninformed': None, 'empirical': None}
    for prior in ['uninformed', 'empirical']:
        # Check if the model has already been learned (and correspondingly 
        # sampled from posterior predictive & generated log-likelihood values)
        if not os.path.exists(f'models/{prior}_model.pkl'):
            model, guide = bm.fit_model(X, regions, y, prior=prior)

            if prior == 'uninformed':
                model_dict[prior] = compute_logprob(model, guide, X, regions, y)
            else:
                μ_a, σ_a = bm.compute_region_params(y)
                μ, τ = bm.compute_feature_params(X, y)
                model_dict[prior] = compute_logprob(model, guide, X, regions, y,
                                                    μ_a, σ_a, μ, τ)

            with open(f'models/{prior}_model.pkl', 'wb') as file:
                dill.dump(model_dict[prior], file)
        else:
            with open(f'models/{prior}_model.pkl', 'rb') as file:
                model_dict[prior] = dill.load(file)
    
    return model_dict
