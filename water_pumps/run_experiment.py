import arviz as az
from compare_models import compare_models
from data_pipeline import prepare_data
import os


if __name__ == '__main__':
    xfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/X.csv'
    yfile = '/home/zblanks/Documents/uva/fall21/bayesian-ml/project/data/y.csv'
    df = prepare_data(xfile, yfile)

    X = df.drop(columns=['working_well', 'region']).to_numpy()
    y = df['working_well'].to_numpy()
    regions = df['region'].to_numpy()

    # Compute the WAIC comparison between the empirical and uninformed models
    model_dict = compare_models(X, regions, y)
    waic_df = az.compare(
        {'uninformed': model_dict['uninformed'], 'empirical': model_dict['empirical']},
        ic='waic', scale='deviance'
    )

    if not os.path.isdir('results'):
        os.mkdir('results')
    
    waic_df.to_csv('results/waic.csv')
