import arviz as az
import bayesian_model as bm
from compare_models import compare_models
from data_pipeline import prepare_data
import dill
from itertools import product
from jax import vmap
import matplotlib.pyplot as plt
import numpy as np
from optimization_model import WaterPointRepair
import os
import pandas as pd
from sklearn.metrics import classification_report


def check_model_performance(X, regions, y, model, threshold=0.5):
    α_hat = np.array(model.posterior['α']).squeeze().mean(axis=0)
    θ_hat = np.array(model.posterior['θ']).squeeze().mean(axis=0)
    β = vmap(bm.compute_logit, in_axes=0)(α_hat[regions], θ_hat[regions], X)

    probs = 1 / (1 + np.exp(-β))

    # Just do a simple rule of > threshold -> label 1 (sufficient as a simple 
    # model check)
    y_hat = (probs > threshold).astype(int)
    return classification_report(y, y_hat, output_dict=True)


def create_prediction_interval(X, regions, model, prob=0.94):
    nsamples = X.shape[0]
    
    if not os.path.exists('results/y_hat.npy'):
        α = np.array(model.posterior['α']).squeeze()
        θ = np.array(model.posterior['θ']).squeeze()

        ndraws = α.shape[0]
        β = np.empty(shape=(ndraws, nsamples))
        for i in range(ndraws):
            α_hat = α[i, :]
            θ_hat = θ[i, :, :]
            β[i, :] = vmap(bm.compute_logit, in_axes=0)(α_hat[regions], θ_hat[regions], X)
        
        y_hat = 1 / (1 + np.exp(-β))
        np.save('results/y_hat.npy', y_hat)
    else:
        y_hat = np.load('results/y_hat.npy')

    lower_bounds = np.empty(shape=(nsamples,))
    for i in range(nsamples):
        lower_bounds[i] = az.hdi(y_hat[:, i], hdi_prob=prob)[0]

    return lower_bounds


def fix_waterpoints(X, regions, model, df, Γs, Bs, region='Dar es Salaam'):
    y_bound = create_prediction_interval(X, regions, model)
    idx = [df.index.get_loc(val) for val in df[df.region == region].index]
    p = df['log_population'].to_numpy()

    rng = np.random.default_rng(17)
    c_bar = rng.uniform(100., 2000., size=len(y_bound))
    δ = np.empty(shape=(len(y_bound),))
    for i in range(len(y_bound)):
        δ[i] = rng.uniform(0., c_bar[i], size=1)

    problem = WaterPointRepair(y_bound[idx], p[idx], c_bar[idx], δ[idx])
    zs = {}
    vals = product(Γs, Bs)
    for val in vals:
        problem.Γ.value = val[0]
        problem.B.value = val[1]
        zs[(val[0], val[1])] = problem.solve()
    
    return zs


if __name__ == '__main__':
    xfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv'
    yfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv'
    df = prepare_data(xfile, yfile)
    drop_cols = ['working_well', 'region_code', 'region', 'latitude',
                 'longitude']

    X = df.drop(columns=drop_cols).to_numpy()
    y = df['working_well'].to_numpy()
    regions = df['region_code'].to_numpy()

    # Compute the WAIC comparison between the empirical and uninformed models
    model_dict = compare_models(X, regions, y)
    waic_df = az.compare(
        {'uninformed': model_dict['uninformed'], 'empirical': model_dict['empirical']},
        ic='waic', scale='deviance'
    )

    if not os.path.isdir('results'):
        os.mkdir('results')
    
    waic_df.to_csv('results/waic.csv')

    # Simple Forest plot comparing the learned interecept distribution for each
    # region
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    az.plot_forest(model_dict['uninformed'], var_names=['α'], ax=axes[0])
    az.plot_forest(model_dict['empirical'], var_names=['α'], ax=axes[1])
    axes[0].set_ylabel('Uninformed Prior', fontsize=20)
    axes[1].set_ylabel('Empirical Prior', fontsize=20)
    plt.savefig('results/forestplot.pdf', dpi=300, bbox_inches='tight')

    report = check_model_performance(X, regions, y, model_dict['uninformed'])
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('results/classification_report.csv')

    # Finally use the prediction bounds to solve the optimization problem
    # for various budgets and degrees of robustness
    if not os.path.exists('results/optimal_repairs.pkl'):
        z_star = fix_waterpoints(X, regions, model_dict['uninformed'], df,
                                 [2, 3], [10000, 100000])
        
        with open('results/optimal_repairs.pkl', 'wb') as file:
            dill.dump(z_star, file)
    else:
        with open('results/optimal_repairs.pkl', 'rb') as file:
            z_star = dill.load(file)
