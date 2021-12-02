import arviz as az
import bayesian_model as bm
from compare_models import compare_models
from data_pipeline import prepare_data
import dill
from itertools import product
from jax import vmap
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from optimization_model import WaterPointRepair
import os
import pandas as pd
import shapefile as shp
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


def get_repair_idx(df, z, region):
    idx = [df.index.get_loc(val) for val in df[df.region == region].index] 
    idx_map = dict(zip(range(len(idx)), idx))
    repair_idx = np.where(z == 1)[0]
    repair_idx = [idx_map[val] for val in repair_idx]
    return repair_idx


def get_region_bounds(shp_file):
    shape = shp_file.shape(1) # DES is the 2nd administrative region
    i0 = shape.parts[2] # DES has two small uninhabitated islands
    i1 = len(shape.points)
    seg = shape.points[i0:(i1 + 1)]
    x = [seg[i][0] for i in range(len(seg))]
    y = [seg[i][1] for i in range(len(seg))]
    return x, y


def plot_repairs(df, y_bound, z_star, key, coords, c, fig=None, ax=None, region='Dar es Salaam'):
    z = z_star[key]
    idx = [df.index.get_loc(val) for val in df[df.region == region].index] 
    repair_idx = get_repair_idx(df, z, region)

    lon = df.iloc[repair_idx].longitude.to_numpy()
    lat = df.iloc[repair_idx].latitude.to_numpy()
    p = df.iloc[repair_idx].log_population.to_numpy()
    y_bound = y_bound[np.where(z == 1)[0]]

    c = c[idx]
    c = c[np.where(z == 1)[0]]

    # To adjust the size of the marker, I'm interested in the ratio of
    # dollar / log person -- i.e., how efficient is this repair
    # for the given location (smaller is better)
    ratio = c / p

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })
    budget_str = '{:,}'.format(key[1])

    ax.plot(*coords, color='black')
    ax.scatter(lon, lat, c=y_bound, s=ratio, alpha=0.5, cmap='winter')
    ax.text(0.6, 0.9, s=rf'$\Gamma = {key[0]}$', size=20, transform=ax.transAxes)
    ax.text(0.6, 0.85, s=rf'$B = \${budget_str}$', size=20, transform=ax.transAxes)
    ax.axis('off')
    return ax


def create_shadow_plot(df, model, col='log_population', region='Dar es Salaam'):
    df = df.query(f'region == "{region}"')

    # We need the index for the region and the feature to compute the various
    # posterior estimates
    col_idx = df.columns.get_loc(col)
    region_idx = df.region_code.unique()

    α = np.array(model.posterior['α']).squeeze()[:, region_idx]
    θ = np.array(model.posterior['θ']).squeeze()[:, region_idx, col_idx]

    # To generate the shadow plot we need to generate  reasonable range of 
    # samples for the col
    X = np.linspace(-0.5, 3, num=500).reshape(1, -1)

    βs = α + (θ @ X)
    logits = 1 / (1 + np.exp(-βs))

    # Also want to show the MAP estimate
    α_map = α.mean()
    θ_map = θ.mean()
    β_map = α_map + (θ_map * X.flatten())
    logit_map = 1 / (1 + np.exp(-β_map)) 

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })

    _, ax = plt.subplots(figsize=(12, 8))
    for i in range(logits.shape[0]):
        ax.plot(X.flatten(), logits[i, :], alpha=0.05, c='black')
    
    ax.plot(X.flatten(), logit_map, c='blue', linestyle='dashed')
    ax.set_xlabel('Normalized Log-Population', fontsize=20)
    ax.set_ylabel(r'$\mathbb{P}$(Working Waterpoint)', fontsize=20)
    ax.set_title('Logistic Posterior Distributions', fontsize=32)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('results/shadowplot.pdf', dpi=300, bbox_inches='tight')


def make_interval_plot(df, region='Dar es Salaam', prob=0.94):
    y_hat = np.load('results/y_hat.npy')

    # Let's get a sample where the waterpoint is broken and where it's
    # functional
    working_samples = df.query(f'region == "{region}" & working_well == 1').index
    broken_samples = df.query(f'region == "{region}" & working_well == 0').index

    working_sample = df.index.get_loc(working_samples[0])
    broken_sample = df.index.get_loc(broken_samples[0])

    y_working = y_hat[:, working_sample]
    y_broken = y_hat[:, broken_sample]

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    az.plot_posterior(y_working, hdi_prob=prob, ax=axes[0])
    az.plot_posterior(y_broken, hdi_prob=prob, ax=axes[1])
    axes[0].set_title('Working Waterpoint Sample', fontsize=20)
    axes[1].set_title('Broken Waterpoint Sample', fontsize=20)
    plt.savefig('results/y-distn.pdf', dpi=300, bbox_inches='tight')


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
                                 [2, 3], [1000, 10000])
        
        with open('results/optimal_repairs.pkl', 'wb') as file:
            dill.dump(z_star, file)
    else:
        with open('results/optimal_repairs.pkl', 'rb') as file:
            z_star = dill.load(file)

    # Using the computed optimal repairs, let's see what the algorithm
    # decided and then plot it on a map
    shp_file = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/tza/tza.shp'
    sf = shp.Reader(shp_file)
    y_bound = create_prediction_interval(X, regions, model_dict['uninformed'])
    des_coords = get_region_bounds(sf)
    rng = np.random.default_rng(17)
    c_bar = rng.uniform(100., 2000., size=X.shape[0])
    keys = [(2, 10000), (3, 10000)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    for i in range(2):
        plot_repairs(
            df=df,
            y_bound=y_bound,
            z_star=z_star,
            key=keys[i],
            coords=des_coords,
            c=c_bar,
            fig=fig,
            ax=axes[i]
        )

    cbar = fig.colorbar(ScalarMappable(cmap='winter'), ax=axes.ravel().tolist())
    cbar.set_label('Probability of Working Waterpoint', fontsize=24)
    plt.savefig('results/repairs.pdf', dpi=300, bbox_inches='tight')

    # Another intersesting visualization is to see the range of posterior 
    # samples for a given feature and region
    create_shadow_plot(df, model_dict['uninformed'])

    # For the last plot let's see what the posterior distribution estimate
    # for a given y_i sample
    make_interval_plot(df)
