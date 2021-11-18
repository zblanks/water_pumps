import numpy as np
import pandas as pd


def compute_date_difference(df: pd.DataFrame) -> pd.DataFrame:
    df.construction_year = pd.to_datetime(df.construction_year, format='%Y')
    df.date_recorded = pd.to_datetime(df.date_recorded, format='%Y/%m/%d')
    df['date_diff'] = (df.date_recorded - df.construction_year).dt.days

    df = (
        df.query('date_diff > 0')
        .assign(log_date_diff = np.log(df.date_diff))
    )

    return df


def prepare_data(Xfile: str, yfile: str) -> pd.DataFrame:
    df = pd.read_csv(Xfile)
    y = pd.read_csv(yfile)

    # Simplifying the problem to binary versus; can generalize in the future
    label_map = {'functional': 1, 'functional needs repair': 1,
                 'non functional': 0}

    payment_map = {'monthly': 1, 'never pay': 0, 'unknown': 0, 'annually': 1,
                   'per bucket': 1, 'on failure': 1, 'other': 1}

    quantity_map = {'dry': 'dry', 'unknown': 'dry', 'enough': 'enough',
                    'seasonal': 'enough', 'insufficient': 'insufficient'}

    df = (
        df.query('longitude != 0' & 'population > 0')
        .query('construction_year != 0')
        .assign(log_population = np.log(df['population']))
        .join(y, on='id', how='inner')
        .rename({'status_group': 'working_well'})
        .replace({'working_well': label_map})
        .replace({'requires_payment': payment_map})
        .replace({'quantity': quantity_map})
        .pipe(compute_date_difference)
    )

    # After investigating in the Pluto notebooks, I'm only going to work with
    # a subset of the columns
    cols = ['id', 'region', 'lga', 'ward', 'quantity', 'source', 
            'log_population', 'waterpoint_type', 'log_date_diff',
            'requires_payment', 'working_well']
    df = df.filter(items=cols)

    # To work with the Bayesian logistic regression model, we must 
    # one-hot-encode the categorical features
    one_hot_features = ['quantity', 'source', 'waterpoint_type']
    df = pd.get_dummies(df, columns=one_hot_features)

    # To avoid the issue of perfect multi-collinearity, I have to remove one
    # column from each of the one-hot-features
    remove_features = ['quantity_dry', 'source_unknown', 'waterpoint_type_other']
    return df.drop(columns=remove_features)
