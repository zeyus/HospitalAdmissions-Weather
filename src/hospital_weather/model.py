import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib  # type: ignore
import logging


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from common.config import DATA_DIR, FIGURE_DIR, read_data, prepare_data





if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Load the data
    # Load the data
    target = 'cov19'
    features = [
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
        target,
    ]

    data = prepare_data(read_data())
    
    # only keep target and features
    data = data[features]
    # Split the data into features and target
    X = data.drop(columns=[target])
    y = data[target]

    # Split the data into train and test, we want to use sliding window (use the last N days to predict the next day)
    split = int(0.8 * len(data))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logging.info(f'Train size: {len(X_train)}')
    logging.info(f'Test size: {len(X_test)}')

    logging.info('Running Random Forest basic model')
    # Train a Random Forest model
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f'Random Forest MSE: {mse:.2f}')
    logging.info(f'Random Forest R2: {r2:.2f}')

    # Save the model
    logging.info('Saving Random Forest model')
    joblib.dump(rf, DATA_DIR / 'random_forest_basic.joblib')

    # plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'predictions_vs_actual_random_forest_basic.png')

    # Train a XGBoost model
    logging.info('Running XGBoost basic model')
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f'XGBoost MSE: {mse:.2f}')
    logging.info(f'XGBoost R2: {r2:.2f}')

    # Save the model
    logging.info('Saving XGBoost model')
    joblib.dump(xgb, DATA_DIR / 'xgboost_basic.joblib')
    
    # plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'predictions_vs_actual_xgboost_basic.png')

    # Plot the feature importance
    logging.info('Plotting feature importance')
    plt.figure(figsize=(10, 5))
    sns.barplot(x=X.columns, y=rf.feature_importances_)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'feature_importance.png')
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    logging.info('Running Random Forest hyperparameter tuning')
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    logging.info(grid_search.best_params_)
    logging.info(grid_search.best_score_)

    # Save the best model
    logging.info('Saving tuned Random Forest model')
    joblib.dump(grid_search.best_estimator_, DATA_DIR / 'random_forest_tuned.joblib')

    # plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, grid_search.best_estimator_.predict(X_test), label='Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'predictions_vs_actual_random_forest_tuned.png')


    # Randomized search
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    logging.info('Running Random Forest randomized search')
    rf = RandomForestRegressor()
    random_search = RandomizedSearchCV(rf, param_dist, cv=5, n_iter=10)
    random_search.fit(X_train, y_train)

    logging.info(random_search.best_params_)
    logging.info(random_search.best_score_)

    # Save the best model
    logging.info('Saving randomized search Random Forest model')
    joblib.dump(random_search.best_estimator_, DATA_DIR / 'random_forest_randomized.joblib')

    # plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, random_search.best_estimator_.predict(X_test), label='Predicted')
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / 'predictions_vs_actual_random_forest_randomized.png')
