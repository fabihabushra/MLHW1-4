
import numpy as np
import argparse
import os
import sys
from dataloader import load_data, load_data_concrete, add_bias_term, handle_unknown_values, process_numerical_attributes, one_hot_encode
from DecisionTree.decisiontree import id3, classify
from EnsembleLearning.adaboost import adaboost, prepare_data as prepare_adaboost_data
from EnsembleLearning.bagging import bagging, prepare_data as prepare_bagging_data, plot_bagging_errors
from EnsembleLearning.randomforest import random_forest, prepare_data as prepare_randomforest_data, plot_random_forest_errors
from EnsembleLearning.bagging_bias_variance import run_bias_variance_decomposition
from EnsembleLearning.random_forest_bias_variance import run_random_forest_bias_variance
from LinearRegression.batch_gradient_descent import batch_gradient_descent
from LinearRegression.stochastic_gradient_descent import stochastic_gradient_descent
from LinearRegression.analytical_solution import analytical_solution

def calculate_error(tree, data, dataset_type="train"):
    correct = 0
    for row in data:
        prediction = classify(tree, row)
        actual_label = row[-1]
        if prediction == actual_label:
            correct += 1
    accuracy = correct / len(data)
    return 1 - accuracy  

def infer_column_types(data, num_check_threshold=5):
    categorical_indices = []
    numerical_indices = []
    
    num_cols = len(data[0])
    for col_idx in range(num_cols - 1):  
        is_numerical = True
        for row in data[:num_check_threshold]:
            try:
                float(row[col_idx])
            except ValueError:
                is_numerical = False
                break
        
        if is_numerical:
            numerical_indices.append(col_idx)
        else:
            categorical_indices.append(col_idx)
    
    return categorical_indices, numerical_indices

def run_experiments_car(train_file, test_file, max_depth):
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    attributes = list(range(len(train_data[0]) - 1))  
    
    results_by_criterion = {
        'information_gain': [],
        'gini': [],
        'majority_error': []
    }

    for depth in range(1, max_depth + 1):        
        for criterion in ['information_gain', 'gini', 'majority_error']:
            tree = id3(train_data, attributes, 0, depth, criterion=criterion)

            train_error = calculate_error(tree, train_data)
            test_error = calculate_error(tree, test_data)

            results_by_criterion[criterion].append({
                "Max Depth": depth,
                "Train Error": train_error,
                "Test Error": test_error
            })
    
    print("\nResults for Car Dataset:\n")
    
    for criterion in ['information_gain', 'gini', 'majority_error']:
        print(f"Results for {criterion.replace('_', ' ').capitalize()} Criterion:")
        print(f"{'Max Depth':<10} {'Train Error':<12} {'Test Error':<12}")
        print("-" * 40)
        for result in results_by_criterion[criterion]:
            print(f"{result['Max Depth']:<10} {result['Train Error']:<12.4f} {result['Test Error']:<12.4f}")
        print("\n")

def run_experiments_bank(train_file, test_file, max_depth, handle_unknown):
    
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    categorical_indices, numerical_indices = infer_column_types(train_data)

    if handle_unknown == 'y':
        train_data = handle_unknown_values(train_data, categorical_indices)
        test_data = handle_unknown_values(test_data, categorical_indices)
    
    train_data, medians = process_numerical_attributes(train_data, numerical_indices)
    test_data, _ = process_numerical_attributes(test_data, numerical_indices, medians)

    attributes = list(range(len(train_data[0]) - 1))  
    results_by_criterion = {
        'information_gain': [],
        'gini': [],
        'majority_error': []
    }

    for depth in range(1, max_depth + 1):        
        for criterion in ['information_gain', 'gini', 'majority_error']:
            tree = id3(train_data, attributes, 0, depth, criterion=criterion, numerical_indices=numerical_indices)

            train_error = calculate_error(tree, train_data, dataset_type="train")
            test_error = calculate_error(tree, test_data, dataset_type="test")

            results_by_criterion[criterion].append({
                "Max Depth": depth,
                "Train Error": train_error,
                "Test Error": test_error
            })
    
    print("\nResults for Bank Dataset:\n")
    
    for criterion in ['information_gain', 'gini', 'majority_error']:
        print(f"Results for {criterion.replace('_', ' ').capitalize()}:")
        print(f"{'Max Depth':<10} {'Train Error':<12} {'Test Error':<12}")
        print("-" * 40)
        for result in results_by_criterion[criterion]:
            print(f"{result['Max Depth']:<10} {result['Train Error']:<12.4f} {result['Test Error']:<12.4f}")
        print("\n")

def run_adaboost_bank(train_file, test_file, iterations, output_directory):
    
    
    X_train, y_train, medians = prepare_adaboost_data(train_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                                     categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15])
    
    X_test, y_test, _ = prepare_adaboost_data(test_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                             categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15], medians=medians)
    
    training_errors, test_errors, stump_training_errors = adaboost(X_train, y_train, X_test, y_test, iterations, output_directory)
    
    print(f"AdaBoost final training error: {training_errors[-1]:.4f}")
    print(f"AdaBoost final test error: {test_errors[-1]:.4f}")

def run_bagging_bank(train_file, test_file, num_trees, output_directory):
    
    
    X_train, y_train, medians = prepare_bagging_data(train_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                                    categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15])
    
    X_test, y_test, _ = prepare_bagging_data(test_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                            categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15], medians=medians)
    
    training_errors, test_errors = bagging(X_train, y_train, X_test, y_test, num_trees)
    
    plot_bagging_errors(training_errors, test_errors, output_directory)
    
    print(f"Bagging final training error: {training_errors[-1]:.4f}")
    print(f"Bagging final test error: {test_errors[-1]:.4f}")

def run_randomforest_bank(train_file, test_file, num_trees, max_features_list, output_directory):
    
    from collections import defaultdict

    
    X_train, y_train, medians = prepare_randomforest_data(train_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                                          categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15])
    
    X_test, y_test, _ = prepare_randomforest_data(test_file, numerical_indices=[0, 5, 9, 11, 12, 13, 14],
                                                  categorical_indices=[1, 2, 3, 4, 6, 7, 8, 10, 15], medians=medians)
    
    errors_dict = defaultdict(tuple)

    for max_features in max_features_list:
        print(f"\nTraining Random Forest with max_features = {max_features}")
        training_errors, test_errors = random_forest(X_train, y_train, X_test, y_test, num_trees, max_features)
        errors_dict[max_features] = (training_errors, test_errors)
        
        print(f"Random Forest final training error with max_features={max_features}: {training_errors[-1]:.4f}")
        print(f"Random Forest final test error with max_features={max_features}: {test_errors[-1]:.4f}")

    
    plot_random_forest_errors(errors_dict, output_directory)

def run_batch_gradient_descent_concrete(train_file, test_file, initial_learning_rate, decay_factor, decay_interval, results_directory):
    
    X_train, y_train = load_data_concrete(train_file)
    X_test, y_test = load_data_concrete(test_file)

    
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)

    
    w = batch_gradient_descent(
        X_train, y_train, X_test, y_test,
        initial_r=initial_learning_rate,
        decay_factor=decay_factor,
        decay_interval=decay_interval,
        results_directory=results_directory
    )

    
    n_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]

    y_pred_train = X_train.dot(w)
    error_train = y_pred_train - y_train
    final_train_cost = (1 / (2 * n_samples)) * np.sum(error_train ** 2)

    y_pred_test = X_test.dot(w)
    error_test = y_pred_test - y_test
    final_test_cost = (1 / (2 * n_test_samples)) * np.sum(error_test ** 2)

    
    print("\nBatch Gradient Descent Results:")
    print(f"Final training cost: {final_train_cost:.4f}")
    print(f"Final test cost: {final_test_cost:.4f}")
    print("\nLearned weight vector (Batch Gradient Descent):")
    print(w)


def run_stochastic_gradient_descent_concrete(train_file, test_file, initial_learning_rate, decay_factor, decay_interval, results_directory):
    
    X_train, y_train = load_data_concrete(train_file)
    X_test, y_test = load_data_concrete(test_file)

    
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)

    
    w = stochastic_gradient_descent(
        X_train, y_train, X_test, y_test,
        initial_r=initial_learning_rate,
        decay_factor=decay_factor,
        decay_interval=decay_interval,
        results_directory=results_directory
    )

    
    n_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]

    y_pred_train = X_train.dot(w)
    error_train = y_pred_train - y_train
    final_train_cost = (1 / (2 * n_samples)) * np.sum(error_train ** 2)

    y_pred_test = X_test.dot(w)
    error_test = y_pred_test - y_test
    final_test_cost = (1 / (2 * n_test_samples)) * np.sum(error_test ** 2)

    
    print("\nStochastic Gradient Descent Results:")
    print(f"Final training cost: {final_train_cost:.4f}")
    print(f"Final test cost: {final_test_cost:.4f}")
    print("\nLearned weight vector (Stochastic Gradient Descent):")
    print(w)


def run_analytical_solution_concrete(train_file, test_file, results_directory):
    
    X_train, y_train = load_data_concrete(train_file)
    X_test, y_test = load_data_concrete(test_file)

    
    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)

    
    w = analytical_solution(X_train, y_train)

    
    n_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]

    y_pred_train = X_train.dot(w)
    error_train = y_pred_train - y_train
    final_train_cost = (1 / (2 * n_samples)) * np.sum(error_train ** 2)

    y_pred_test = X_test.dot(w)
    error_test = y_pred_test - y_test
    final_test_cost = (1 / (2 * n_test_samples)) * np.sum(error_test ** 2)

    
    print("\nAnalytical Solution Results:")
    print(f"Final training cost: {final_train_cost:.4f}")
    print(f"Final test cost: {final_test_cost:.4f}")
    print("\nOptimal weight vector (Analytical Solution):")
    print(w)

def main():
    parser = argparse.ArgumentParser(description='Run machine learning algorithms on datasets.')
    parser.add_argument('--algorithm', type=str, required=True, choices=['decisiontree', 'adaboost', 'bagging', 'randomforest', 'bagging_bias_variance', 'random_forest_bias_variance', 'batch_gradient_descent', 'stochastic_gradient_descent', 'analytical_solution'],
                        help='Algorithm to run')
    parser.add_argument('--dataset', type=str, required=True, choices=['car', 'bank', 'concrete'],
                        help='Dataset to use')

    
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree')

    
    parser.add_argument('--iterations', type=int, default=500, help='Number of iterations (for AdaBoost, Bagging, RandomForest)')

    
    parser.add_argument('--max_features', type=str, default='4', help='Comma-separated list of max features to consider at each split (for Random Forest)')

    
    parser.add_argument('--initial_learning_rate', type=float, default=0.01, help='Initial learning rate for gradient descent methods')
    parser.add_argument('--decay_factor', type=float, default=0.8, help='Decay factor for learning rate')
    parser.add_argument('--decay_interval', type=int, default=500, help='Number of iterations between learning rate decay')

    
    parser.add_argument('--handle_unknown', type=str, choices=['y', 'n'], default='n',
                        help="Do you want to replace 'unknown' values with the majority of other attribute values in the training set? (y/n)")
    parser.add_argument('--num_runs', type=int, default=100, help='Number of runs for bias-variance decomposition')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size for each run')
    parser.add_argument('--num_trees', type=int, default=500, help='Number of trees in the ensemble')
    parser.add_argument('--bias_variance_max_features', type=int, default=4,
                    help='Number of features to consider at each split (for Random Forest)')



    args = parser.parse_args()

    algorithm = args.algorithm
    dataset = args.dataset

    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, dataset, 'train.csv')
    test_file = os.path.join(base_dir, dataset, 'test.csv')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Dataset files not found for dataset '{dataset}'. Expected files: {train_file}, {test_file}")
        sys.exit(1)

    
    results_directory = os.path.join(base_dir, 'results')
    os.makedirs(results_directory, exist_ok=True)

    

    if algorithm == 'decisiontree':
        max_depth = args.max_depth
        if dataset == 'car':
            run_experiments_car(train_file, test_file, max_depth)
        elif dataset == 'bank':
            run_experiments_bank(train_file, test_file, max_depth, args.handle_unknown)
        else:
            print(f"Decision Tree not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'adaboost':
        iterations = args.iterations
        if dataset == 'bank':
            run_adaboost_bank(train_file, test_file, iterations, results_directory)
        else:
            print(f"AdaBoost not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'bagging':
        num_trees = args.iterations  
        if dataset == 'bank':
            run_bagging_bank(train_file, test_file, num_trees, results_directory)
        else:
            print(f"Bagging not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'randomforest':
        num_trees = args.iterations  
        max_features_list = [int(x) for x in args.max_features.split(',')]
        if dataset == 'bank':
            run_randomforest_bank(train_file, test_file, num_trees, max_features_list, results_directory)
        else:
            print(f"Random Forest not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'bagging_bias_variance':
        if dataset == 'bank':
            run_bias_variance_decomposition(train_file, test_file, results_directory)
        else:
            print(f"Bias-Variance Decomposition not implemented for dataset '{dataset}'")
            sys.exit(1)

    elif algorithm == 'random_forest_bias_variance':
        if dataset == 'bank':
            run_random_forest_bias_variance(train_file, test_file, results_directory,
                                            num_runs=args.num_runs, sample_size=args.sample_size,
                                            num_trees=args.num_trees, max_features=args.bias_variance_max_features)
        else:
            print(f"Random Forest Bias-Variance Decomposition not implemented for dataset '{dataset}'")
            sys.exit(1)

    elif algorithm == 'batch_gradient_descent':
        if dataset == 'concrete':
            run_batch_gradient_descent_concrete(train_file, test_file, args.initial_learning_rate, args.decay_factor, args.decay_interval, results_directory)
        else:
            print(f"Batch Gradient Descent not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'stochastic_gradient_descent':
        if dataset == 'concrete':
            run_stochastic_gradient_descent_concrete(train_file, test_file, args.initial_learning_rate, args.decay_factor, args.decay_interval, results_directory)
        else:
            print(f"Stochastic Gradient Descent not implemented for dataset '{dataset}'")
            sys.exit(1)
    elif algorithm == 'analytical_solution':
        if dataset == 'concrete':
            run_analytical_solution_concrete(train_file, test_file, results_directory)
        else:
            print(f"Analytical Solution not implemented for dataset '{dataset}'")
            sys.exit(1)
    else:
        print(f"Algorithm '{algorithm}' is not recognized.")
        sys.exit(1)

if __name__ == '__main__':
    main()
