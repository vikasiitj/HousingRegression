import codecs

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Load Boston Housing dataset manually"""
    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is our target variable

    return df


def explore_data(df):
    """Perform basic data exploration"""
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return correlation_matrix


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the data for modeling"""
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")

    return {
        'model_name': model_name,
        'mse': mse,
        'r2': r2,
        'rmse': np.sqrt(mse),
        'predictions': y_pred
    }


def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def get_baseline_models():
    """Return dictionary of baseline models"""
    models = {
        'Lasso Regression': Lasso(random_state=42),
        'Ridge Regression': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    return models


def get_hyperparameter_grids():
    """Return hyperparameter grids for tuning"""
    param_grids = {
        'Lasso Regression': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'max_iter': [1000, 2000, 5000],
            'selection': ['cyclic', 'random']
        },
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky'],
            'max_iter': [1000, 2000, 3000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    return param_grids


def perform_hyperparameter_tuning(models, param_grids, X_train, y_train, cv=5):
    """Perform hyperparameter tuning using GridSearchCV"""
    tuned_models = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")

        if name in param_grids and param_grids[name]:
            grid_search = GridSearchCV(
                model, param_grids[name],
                cv=cv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)

            tuned_models[name] = grid_search.best_estimator_
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.4f}")
        else:
            # For models without hyperparameters (like basic Linear Regression)
            model.fit(X_train, y_train)
            tuned_models[name] = model
            print(f"No hyperparameters to tune for {name}")

    return tuned_models


def save_results_to_file(results, filename):
    """Save results to a text file"""
    with codecs.open(filename, 'w', 'utf-8') as f:
        f.write("Model Performance Results\n")
        f.write("=" * 50 + "\n\n")

        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"MSE: {result['mse']:.4f}\n")
            f.write(f"R²: {result['r2']:.4f}\n")
            f.write(f"RMSE: {result['rmse']:.4f}\n")
            f.write("-" * 30 + "\n")


def compare_models(results):
    """Compare model performances"""
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'MSE': r['mse'],
            'R²': r['r2'],
            'RMSE': r['rmse']
        }
        for r in results
    ])

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # MSE comparison
    axes[0].bar(results_df['Model'], results_df['MSE'])
    axes[0].set_title('Model Comparison - MSE')
    axes[0].set_ylabel('MSE')
    axes[0].tick_params(axis='x', rotation=45)

    # R² comparison
    axes[1].bar(results_df['Model'], results_df['R²'])
    axes[1].set_title('Model Comparison - R²')
    axes[1].set_ylabel('R² Score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results_df
