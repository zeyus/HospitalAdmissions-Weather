import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging
from typing import Any
from tqdm import tqdm

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import root_mean_squared_error, r2_score  # type: ignore
from sklearn.linear_model import LinearRegression, Lasso  # type: ignore
from sklearn.linear_model._base import LinearModel  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore
import matplotlib.dates as mdates


from .common.config import FIGURE_DIR, read_data, prepare_data, MODEL_DIR, SELECTED_TARGET, SELECTED_FEATURES


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
    

def model_eval(
        model: LinearModel,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,
        model_name: str,
        const_pred: None | float = None,
        unscale: bool = False,
        scaler_scale: float = 1.0,
        scaler_min: float = 0.0) -> dict[str, Any]:
    if const_pred is not None:
        print(const_pred)
        if unscale:
            const_pred = const_pred * 1/scaler_scale + scaler_min
        print(const_pred)
        y_pred = np.full_like(y, const_pred)

    else:
        y_pred = model.predict(X)
    if unscale:
        y = y * 1/scaler_scale + scaler_min
        y_pred = y_pred * 1/scaler_scale + scaler_min

    rmse = root_mean_squared_error(y, y_pred)
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
    target = SELECTED_TARGET

    features = SELECTED_FEATURES

    data = data[features]

    logging.info("Scaling data")
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_index = data.columns.get_loc(target)
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    
    
    logging.info("Splitting dataset")
    X_train, X_test, y_train, y_test = prepare_dataset(data, target, test_size=0.3, val_size=0.0)


    # Display covariance matrix of features
    plt.figure()
    sns.heatmap(
        data.corr(),
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
    for x,y,split in tqdm(zip([X_train, X_test], [y_train, y_test], ["train", "test"])):
        model_evals.append(
            model_eval(
                LinearRegression(),
                x,  # type: ignore
                y,
                split,
                "dummy",
                const_pred=y_train.mean(),
                unscale=True,
                scaler_scale=scaler.scale_[target_index],  # type: ignore
                scaler_min=scaler.min_[target_index]  # type: ignore
            ))
        

    
    logging.info("Fitting linear regression model")
    logging.info(f"Using features: {features}")
    model = LinearRegression()
    model.fit(X_train, y_train)
    models['linear regression-0'] = model
    
    logging.info("Evaluating linear regression model")
    for x,y,split in tqdm(zip([X_train, X_test], [y_train, y_test], ["train", "test"])):
        model_evals.append(
            model_eval(
                model,
                x,  # type: ignore
                y,
                split,
                "linear regression-0",
                unscale=True,
                scaler_scale=scaler.scale_[target_index],  # type: ignore
                scaler_min=scaler.min_[target_index]  # type: ignore

            ))
        
    for alpha in [0.00001, 0.0001, 0.01, 0.1, 0.5]:
        logging.info(f"Running Lasso and Ridge regression with alpha={alpha}")
        id = 'lasso'
        reg = Lasso(alpha=alpha).fit(X_train, y_train)
        models[f'{id}-{alpha}'] = reg
        for x,y,nsplit in tqdm(zip([X_train, X_test],
                            [y_train, y_test],
                            ['train', 'test']), desc=f'{id}-{alpha}'):
            model_evals.append(
                model_eval(
                    model=reg, 
                    X=x,  # type: ignore
                    y=y, 
                    split_name=nsplit, 
                    model_name=f'linear-{id}-alpha-{alpha}',
                    unscale=True,
                    scaler_scale=scaler.scale_[target_index],  # type: ignore
                    scaler_min=scaler.min_[target_index]  # type: ignore
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
        columns=data.drop(target, axis=1).columns,
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

    # plot train+test predictions from best model
    # drop dummy model from eval results
    eval_results = eval_results[eval_results['model'] != 'dummy']
    best_model = eval_results.loc[eval_results['rmse'].idxmin()]
    date_formater = mdates.DateFormatter('%b, %Y')
    plt.figure(figsize=(20, 15))
    plt.rcParams.update({'font.size': 15})
    target_index = data.columns.get_loc(target)
    plt.scatter(data.index, data[target] * 1/scaler.scale_[target_index] + scaler.min_[target_index], label='Actual', color='gray')
    plt.plot(data.index[:len(X_train)], models[best_model['model']].predict(X_train) * 1/scaler.scale_[target_index] + scaler.min_[target_index], label='Train Prediction')  # type: ignore
    plt.plot(data.index[len(X_train):], models[best_model['model']].predict(X_test) * 1/scaler.scale_[target_index] + scaler.min_[target_index], label='Test Prediction')  # type: ignore
    # set x tick labels to formated Month, Year
    plt.gca().xaxis.set_major_formatter(date_formater)
    plt.legend()
    plt.savefig(FIGURE_DIR / 'linear_regression_predictions.png', dpi=300, bbox_inches = "tight")
    
