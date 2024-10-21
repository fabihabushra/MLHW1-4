

import numpy as np
from collections import Counter
from DecisionTree.decisiontree import DecisionTree
from EnsembleLearning.bagging import bagging_predictor
from dataloader import prepare_data_bias_variance

def compute_bias_variance(y_true, predictions):
    
    avg_predictions = np.mean(predictions, axis=0)
    bias_squared = (avg_predictions - y_true) ** 2
    variance = np.mean((predictions - avg_predictions) ** 2, axis=0)
    return bias_squared, variance

def run_bias_variance_decomposition(train_file, test_file, results_directory,
                                    num_runs=100, sample_size=1000, num_trees=500):
    
    
    numerical_indices = [0, 5, 9, 11, 12, 13, 14]
    categorical_indices = [1, 2, 3, 4, 6, 7, 8, 10, 15]

    
    X_train_full, y_train_full, medians = prepare_data_bias_variance(train_file, numerical_indices, categorical_indices)

    
    X_test, y_test, _ = prepare_data_bias_variance(test_file, numerical_indices, categorical_indices, medians)

    y_test = np.array(y_test)

    
    num_runs = 100
    sample_size = 1000
    num_trees = 500

    
    single_tree_predictions = []
    bagged_predictions = []

    for run in range(1, num_runs + 1):
        
        indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)
        X_train = [X_train_full[i] for i in indices]
        y_train = [y_train_full[i] for i in indices]

        
        bagged_trees = bagging_predictor(X_train, y_train, num_trees=num_trees)

        
        first_tree = bagged_trees[0]
        single_tree_pred = first_tree.predict(X_test)
        single_tree_predictions.append(single_tree_pred)

        
        
        bagged_pred = np.zeros(len(X_test))
        for tree in bagged_trees:
            pred = tree.predict(X_test)
            bagged_pred += pred
        bagged_pred = np.sign(bagged_pred)  
        bagged_predictions.append(bagged_pred)

        if run % 10 == 0 or run == 1 or run == num_runs:
            print(f"Completed run {run}/{num_runs}")

    
    single_tree_predictions = np.array(single_tree_predictions)
    bagged_predictions = np.array(bagged_predictions)

    
    bias_squared_single, variance_single = compute_bias_variance(y_test, single_tree_predictions)
    mean_bias_single = np.mean(bias_squared_single)
    mean_variance_single = np.mean(variance_single)
    total_error_single = mean_bias_single + mean_variance_single

    
    bias_squared_bagging, variance_bagging = compute_bias_variance(y_test, bagged_predictions)
    mean_bias_bagging = np.mean(bias_squared_bagging)
    mean_variance_bagging = np.mean(variance_bagging)
    total_error_bagging = mean_bias_bagging + mean_variance_bagging

    
    print("\nBias-Variance Decomposition Results:")
    print("Single Decision Tree Learner:")
    print(f"Average Bias^2: {mean_bias_single:.4f}")
    print(f"Average Variance: {mean_variance_single:.4f}")
    print(f"Total Error (Bias^2 + Variance): {total_error_single:.4f}")

    print("\nBagged Trees:")
    print(f"Average Bias^2: {mean_bias_bagging:.4f}")
    print(f"Average Variance: {mean_variance_bagging:.4f}")
    print(f"Total Error (Bias^2 + Variance): {total_error_bagging:.4f}")

    
    results_output_path = os.path.join(results_directory, 'bias_variance_results.txt')
    with open(results_output_path, 'w') as f:
        f.write("Bias-Variance Decomposition Results:\n")
        f.write("Single Decision Tree Learner:\n")
        f.write(f"Average Bias^2: {mean_bias_single:.4f}\n")
        f.write(f"Average Variance: {mean_variance_single:.4f}\n")
        f.write(f"Total Error (Bias^2 + Variance): {total_error_single:.4f}\n\n")
        f.write("Bagged Trees:\n")
        f.write(f"Average Bias^2: {mean_bias_bagging:.4f}\n")
        f.write(f"Average Variance: {mean_variance_bagging:.4f}\n")
        f.write(f"Total Error (Bias^2 + Variance): {total_error_bagging:.4f}\n")

    print(f"\nResults saved to {results_output_path}")
