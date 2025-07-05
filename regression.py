import pandas as pd
import numpy as np
from utils import (
    load_data, explore_data, preprocess_data,
    evaluate_model, plot_predictions, get_baseline_models,
    get_hyperparameter_grids, perform_hyperparameter_tuning,
    save_results_to_file, compare_models
)


def main():
    """Main function to run regression analysis with hyperparameter tuning"""
    print("Loading Boston Housing Dataset...")
    df = load_data()

    print("\nExploring the dataset...")
    correlation_matrix = explore_data(df)

    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("\nTraining baseline models...")
    models = get_baseline_models()
    baseline_results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Evaluate model
        result = evaluate_model(model, X_test, y_test, f"{name} (Baseline)")
        baseline_results.append(result)

    print("\nPerforming hyperparameter tuning...")
    param_grids = get_hyperparameter_grids()
    tuned_models = perform_hyperparameter_tuning(models, param_grids, X_train, y_train)

    print("\nEvaluating tuned models...")
    tuned_results = []

    for name, model in tuned_models.items():
        result = evaluate_model(model, X_test, y_test, f"{name} (Tuned)")
        tuned_results.append(result)

        # Plot predictions for tuned models
        plot_predictions(y_test, result['predictions'], f"{name} (Tuned)")

    # Combine all results
    all_results = baseline_results + tuned_results

    # Compare models
    results_df = compare_models(all_results)

    # Save results
    save_results_to_file(all_results, 'hypertuned_results.txt')

    print("\nHyperparameter tuning completed!")
    print("Check the generated plots and hypertuned_results.txt for detailed results.")


if __name__ == "__main__":
    main()
