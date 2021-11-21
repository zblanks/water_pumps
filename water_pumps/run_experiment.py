from bayesian_model import fit_model
from data_pipeline import prepare_data
from jax import random
from numpyro.infer import Predictive


if __name__ == '__main__':
    xfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv'
    yfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv'
    df = prepare_data(xfile, yfile)

    X = df.drop(columns=['working_well', 'region']).to_numpy()
    y = df['working_well'].to_numpy()
    regions = df['region'].to_numpy()

    model, guide = fit_model(X[:1000, :], regions[:1000], y[:1000])
    predictive = Predictive(guide, params=model.params, num_samples=50)
    results = predictive(random.PRNGKey(17))
    print(results)
