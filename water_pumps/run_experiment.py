from bayesian_model import fit_model
from data_pipeline import prepare_data


if __name__ == '__main__':
    xfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv'
    yfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv'
    df = prepare_data(xfile, yfile)

    # Start with a simple experiment with only two features and a small 
    # sample size for testing
    cols = ['log_date_diff', 'requires_payment']
    X = df.filter(items=cols).to_numpy()
    y = df['working_well'].to_numpy()
    regions = df['region'].to_numpy()

    fit_model(X[:300, :], regions[:300], y[:300])
