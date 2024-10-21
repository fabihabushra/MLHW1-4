

import numpy as np
import random
import os
from collections import Counter
from dataloader import prepare_data_bias_variance
from DecisionTree.decisiontree import DecisionTree
from EnsembleLearning.randomforest import RandomForestTree

def compute_bias_variance(y_true, predictions):
    
    predictions = np.array(predictions)  
    avg_predictions = np.mean(predictions, axis=0)
    bias_squared = (avg_predictions - y_true) ** 2
    variance = np.mean((predictions - avg_predictions) ** 2, axis=0)
    return bias_squared, variance

def run_random_forest_bias_variance(train_file, test_file, results_directory,
                                    num_runs=100, sample_size=1000, num_trees=500, max_features=4):
    
    
    numerical_indices = [0, 5, 9, 11, 12, 13, 14]
    categorical_indices = [1, 2, 3, 4, 6, 7, 8, 10, 15]

    
    X_train_full, y_train_full, medians = prepare_data_bias_variance(train_file, numerical_indices, categorical_indices)

    
    X_test, y_test, _ = prepare_data_bias_variance(test_file, numerical_indices, categorical_indices, medians)

    y_test = np.array(y_test)

    
    single_tree_predictions_rf = []  
    random_forest_predictions = []   

    print("Starting bias-variance decomposition experiment for Random Forest...")

    for run in range(1, num_runs + 1):
        
        indices = np.random.choice(len(X_train_full), size=sample_size, replace=False)
        X_train = [X_train_full[i] for i in indices]
        y_train = [y_train_full[i] for i in indices]

        
        trees = []
        for _ in range(num_trees):
            
            bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap = [X_train[i] for i in bootstrap_indices]
            y_bootstrap = [y_train[i] for i in bootstrap_indices]

            
            tree = RandomForestTree(max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            trees.append(tree)

        
        first_tree = trees[0]
        single_tree_pred = first_tree.predict(X_test)
        single_tree_predictions_rf.append(single_tree_pred)

        
        
        rf_pred = np.zeros(len(X_test))
        for tree in trees:
            pred = tree.predict(X_test)
            rf_pred += pred
        rf_pred = np.sign(rf_pred)  
        random_forest_predictions.append(rf_pred)

        if run % 10 == 0 or run == 1 or run == num_runs:
            print(f"Completed run {run}/{num_runs}")

    
    single_tree_predictions_rf = np.array(single_tree_predictions_rf)
    random_forest_predictions = np.array(random_forest_predictions)

    
    bias_squared_single_rf, variance_single_rf = compute_bias_variance(y_test, single_tree_predictions_rf)
    mean_bias_single_rf = np.mean(bias_squared_single_rf)
    mean_variance_single_rf = np.mean(variance_single_rf)
    total_error_single_rf = mean_bias_single_rf + mean_variance_single_rf

    
    bias_squared_rf, variance_rf = compute_bias_variance(y_test, random_forest_predictions)
    mean_bias_rf = np.mean(bias_squared_rf)
    mean_variance_rf = np.mean(variance_rf)
    total_error_rf = mean_bias_rf + mean_variance_rf

    
    print("\nBias-Variance Decomposition Results for Random Forest:")
    print("Single Random Tree Learner:")
    print(f"Average Bias^2: {mean_bias_single_rf:.4f}")
    print(f"Average Variance: {mean_variance_single_rf:.4f}")
    print(f"Total Error (Bias^2 + Variance): {total_error_single_rf:.4f}")

    print("\nRandom Forest:")
    print(f"Average Bias^2: {mean_bias_rf:.4f}")
    print(f"Average Variance: {mean_variance_rf:.4f}")
    print(f"Total Error (Bias^2 + Variance): {total_error_rf:.4f}")

    
    results_output_path = os.path.join(results_directory, 'bias_variance_results_random_forest.txt')
    with open(results_output_path, 'w') as f:
        f.write("Bias-Variance Decomposition Results for Random Forest:\n")
        f.write("Single Random Tree Learner:\n")
        f.write(f"Average Bias^2: {mean_bias_single_rf:.4f}\n")
        f.write(f"Average Variance: {mean_variance_single_rf:.4f}\n")
        f.write(f"Total Error (Bias^2 + Variance): {total_error_single_rf:.4f}\n\n")
        f.write("Random Forest:\n")
        f.write(f"Average Bias^2: {mean_bias_rf:.4f}\n")
        f.write(f"Average Variance: {mean_variance_rf:.4f}\n")
        f.write(f"Total Error (Bias^2 + Variance): {total_error_rf:.4f}\n")

    print(f"\nResults saved to {results_output_path}")
