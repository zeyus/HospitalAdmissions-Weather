import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from typing import Any
from tqdm import tqdm

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # type: ignore
from sklearn.linear_model._base import LinearModel  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


from .common.config import FIGURE_DIR, read_data, prepare_data, MODEL_DIR


def prepare_dataset(
        data: pd.DataFrame,
        target: str,
        test_size: float = 0.15,
        val_size: float = 0.0,
        random_state: int = 666
        ) -> tuple[np.ndarray, ..., ]:
    X = data.drop(target, axis=1).to_numpy()
    y = data[target].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_test, y_train, y_test
    

def model_eval(model: LinearModel, X: np.ndarray, y: np.ndarray, split_name: str, model_name: str, const_pred: None | float = None) -> dict[str, Any]:
    if const_pred is not None:
        y_pred = np.full_like(y, const_pred)
    else:
        y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return {
        "model": model_name,
        "split": split_name,
        "const_pred": const_pred,
        "rmse": rmse,
        "r2": r2,
    }
    
   

def train_lr() -> None:
    sns.set_style("whitegrid")
    logging.info("Loading data")
    data = prepare_data(read_data())
    target = 'cov19'
    # all_features = [
    #     target,
    #     'humidity_mean',
    #     'humidity_min',
    #     'humidity_max',
    #     'humidity_std',
    #     'precip_sum',
    #     'precip_max',
    #     'pressure_mean',
    #     'pressure_min',
    #     'pressure_max',
    #     'pressure_std',
    #     'temp_mean',
    #     'temp_min',
    #     'temp_max',
    #     'temp_std',
    #     'windgust_mean',
    #     'windgust_max',
    #     'windgust_std',
    #     'windspeed_mean',
    #     'windspeed_max',
    #     'windspeed_std',
    #     'winddir_sin',
    #     'winddir_cos',
    #     'winddir_mean',
    #     'snow_max',
    #     'snowdepth_max',
    #     'solarradiation_max',
    #     'day_of_week',
    # ]
    all_features = [
        target,
        'precip_sum', 
        'pressure_mean',
        'pressure_std', 
        'temp_mean',
        'temp_std',
        'windspeed_mean',
        'windspeed_std',
        'winddir_sin',
        'winddir_cos',
        'snowdepth_max',
    ]

    data = data[all_features]

    
    
    logging.info("Splitting dataset")
    X_train, X_test, X_val, y_train, y_test, y_val = prepare_dataset(data, target, test_size=0.15, val_size=0.15)

    logging.info("Scaling data")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Display covariance matrix of features
    plt.figure()
    sns.heatmap(
        pd.DataFrame(X_train, columns=all_features[1:]).corr(),
        cmap='coolwarm',
        center=0,
    )
    plt.savefig(FIGURE_DIR / 'feature_correlation.png', dpi=300, bbox_inches = "tight")

    # clear previous figure
    plt.clf()

    logging.info("Running regression to identify important features")

    model_evals: list[dict[str, Any]] = []
    models: dict[str, Lasso] = {}

    logging.info("Evaluating dummy model")
    for x,y,split in tqdm(zip([X_train, X_val, X_test], [y_train, y_val, y_test], ["train", "val", "test"])):
        model_evals.append(
            model_eval(
                LinearRegression(),
                x,  # type: ignore
                y,
                split,
                "dummy",
                const_pred=y_train.mean()
            ))
        

    
    logging.info("Fitting linear regression model")
    logging.info(f"Using features: {all_features[1:]}")
    model = LinearRegression()
    model.fit(X_train, y_train)

    logging.info("Evaluating linear regression model")
    for x,y,split in tqdm(zip([X_train, X_val, X_test], [y_train, y_val, y_test], ["train", "val", "test"])):
        model_evals.append(
            model_eval(
                model,
                x,  # type: ignore
                y,
                split,
                "linear regression"
            ))
        
    for alpha in [0.1, 0.01, 0.5, 1.0, 2.0, 5.0]:
        logging.info(f"Running Lasso and Ridge regression with alpha={alpha}")
        id = 'lasso'
        reg = Lasso(alpha=alpha).fit(X_train, y_train)
        models[f'{id}-{alpha}'] = reg
        for x,y,nsplit in tqdm(zip([X_train, X_val, X_test],
                            [y_train, y_val, y_test],
                            ['train', 'val', 'test']), desc=f'{id}-{alpha}'):
            model_evals.append(
                model_eval(
                    model=reg, 
                    X=x,  # type: ignore
                    y=y, 
                    split_name=nsplit, 
                    model_name=f'linear-{id}-alpha-{alpha}'
                )
            )
    

    logging.info("Saving model evaluation results")
    eval_results = pd.DataFrame(model_evals)
    logging.info(eval_results.head())
    eval_results.to_csv(MODEL_DIR / 'linear_regression_eval_results.csv')

    logging.info("Plotting results")
    plt.figure()
    sns.scatterplot(
        data=eval_results.sort_values(by='rmse', ascending=False),
        x='rmse',
        y='model',
        marker='s',
        hue='split',
        palette='viridis',
    )
    plt.legend(loc='upper right')
    plt.savefig(FIGURE_DIR / 'linear_regression_eval_results.png', dpi=300, bbox_inches = "tight")


    logging.info("Plotting coefficients")
    coefs = pd.DataFrame(
        np.vstack([v.coef_.round(4) for v in models.values()]),
        columns=all_features[1:],
    )

    coefs['model_type'] = [k.split('-')[0] for k in models.keys()]
    coefs['alpha'] = [k.split('-')[1] for k in models.keys()]
    coefs = pd.melt(coefs, id_vars=['model_type', 'alpha'])
    coefs['abs'] = coefs['value'].abs()
    print(coefs)
    # now print top 5 by model and alpha
    top_2 = coefs.groupby(['model_type', 'alpha']).apply(lambda x: x.nlargest(2, 'abs'))
    print(top_2)
    # now print bottom 5 by model and alpha
    bottom_2 = coefs.groupby(['model_type', 'alpha']).apply(lambda x: x.nsmallest(2, 'abs'))
    print(bottom_2)
    # now find the most common features in the top 5
    top_2_features = top_2['variable'].value_counts()
    print(top_2_features)
    # now find the most common features in the bottom 5
    bottom_2_features = bottom_2['variable'].value_counts()
    print(bottom_2_features)

    # save coefficients
    coefs.to_csv(MODEL_DIR / 'linear_regression_coefficients.csv')

    plt.figure()
    grid = sns.relplot(
        data=coefs[coefs['model_type'].isin(['lasso', 'ridge'])],
        x='alpha',
        y='value',
        hue='variable',
        col='model_type',
        kind='line',
    )
    # grid.set(xscale='log')
    grid.set_axis_labels('coefficient', 'feature')
    plt.savefig(FIGURE_DIR / 'linear_regression_coefficients.png', dpi=300, bbox_inches = "tight")


    
