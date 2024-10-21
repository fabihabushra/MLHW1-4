

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import os

from dataloader import load_data, process_numerical_attributes, one_hot_encode

def prepare_data(filename, numerical_indices, categorical_indices, medians=None):
    
    data = load_data(filename)
    data, medians = process_numerical_attributes(data, numerical_indices, medians)
    labels = [1 if row[-1].lower() == 'yes' else -1 for row in data]
    data = [row[:-1] for row in data]
    data = one_hot_encode(data, categorical_indices)
    return data, labels, medians

def calculate_weighted_entropy(y, sample_weights):
    
    total_weight = np.sum(sample_weights)
    entropy = 0
    for label in np.unique(y):
        label_weight = np.sum(sample_weights[np.array(y) == label])
        probability = label_weight / total_weight
        if probability > 0:
            entropy -= probability * np.log2(probability)
    return entropy

def calculate_weighted_information_gain(y, y_left, y_right, sample_weights, left_weights, right_weights):
    
    entropy_before = calculate_weighted_entropy(y, sample_weights)
    total_weight = np.sum(sample_weights)
    weight_left = np.sum(left_weights)
    weight_right = np.sum(right_weights)
    entropy_left = calculate_weighted_entropy(y_left, left_weights)
    entropy_right = calculate_weighted_entropy(y_right, right_weights)
    entropy_after = (weight_left / total_weight) * entropy_left + (weight_right / total_weight) * entropy_right
    information_gain = entropy_before - entropy_after
    return information_gain

def decision_stump(X, y, sample_weights):
    
    m, n = len(X), len(X[0])
    max_info_gain = -float('inf')
    best_stump = {}
    X = np.array(X)
    y = np.array(y)
    sample_weights = np.array(sample_weights)

    for feature_i in range(n):
        values = np.unique(X[:, feature_i])
        for threshold in values:
            
            indices_left = X[:, feature_i] == threshold
            indices_right = X[:, feature_i] != threshold

            y_left = y[indices_left]
            y_right = y[indices_right]
            left_weights = sample_weights[indices_left]
            right_weights = sample_weights[indices_right]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            info_gain = calculate_weighted_information_gain(y, y_left, y_right, sample_weights, left_weights, right_weights)

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stump['feature_index'] = feature_i
                best_stump['threshold'] = threshold
                best_stump['indices_left'] = indices_left
                best_stump['indices_right'] = indices_right

    if not best_stump:
        
        majority_label = Counter(y).most_common(1)[0][0]
        best_stump['feature_index'] = -1
        best_stump['threshold'] = None
        best_stump['indices_left'] = np.array([False]*m)
        best_stump['indices_right'] = np.array([False]*m)
        best_stump['left_prediction'] = majority_label
        best_stump['right_prediction'] = majority_label
        best_stump['predictions'] = [majority_label] * m
        return best_stump

    
    predictions = np.zeros(m)
    left_labels = y[best_stump['indices_left']]
    right_labels = y[best_stump['indices_right']]
    left_weighted_majority = np.sign(np.sum(sample_weights[best_stump['indices_left']] * left_labels))
    right_weighted_majority = np.sign(np.sum(sample_weights[best_stump['indices_right']] * right_labels))
    
    left_weighted_majority = left_weighted_majority if left_weighted_majority != 0 else 1
    right_weighted_majority = right_weighted_majority if right_weighted_majority != 0 else 1
    predictions[best_stump['indices_left']] = left_weighted_majority
    predictions[best_stump['indices_right']] = right_weighted_majority

    best_stump['predictions'] = predictions.tolist()
    best_stump['left_prediction'] = left_weighted_majority
    best_stump['right_prediction'] = right_weighted_majority

    return best_stump

def predict(X, classifiers, alphas):
    
    m = len(X)
    final_predictions = np.zeros(m)
    X = np.array(X)
    for alpha, stump in zip(alphas, classifiers):
        if stump['feature_index'] == -1:
            
            predictions = np.full(m, stump['left_prediction'])
        else:
            feature_i = stump['feature_index']
            threshold = stump['threshold']
            indices_left = X[:, feature_i] == threshold
            indices_right = X[:, feature_i] != threshold
            predictions = np.zeros(m)
            predictions[indices_left] = stump['left_prediction']
            predictions[indices_right] = stump['right_prediction']
        final_predictions += alpha * predictions
    return np.sign(final_predictions)

def calculate_weighted_error(y_true, y_pred, sample_weights):
    
    incorrect = (np.array(y_true) != np.array(y_pred)).astype(int)
    weighted_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)
    return weighted_error

def adaboost(X_train, y_train, X_test, y_test, T, output_directory):
    
    m = len(X_train)
    sample_weights = np.full(m, (1 / m))
    classifiers = []
    alphas = []
    training_errors = []
    test_errors = []
    stump_training_errors = []

    for t in range(1, T + 1):
        stump = decision_stump(X_train, y_train, sample_weights)
        y_pred = stump['predictions']
        epsilon = calculate_weighted_error(y_train, y_pred, sample_weights)
        if epsilon == 0:
            epsilon = 1e-10  
        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        alphas.append(alpha)
        classifiers.append(stump)
        
        
        y_train_pred = predict(X_train, classifiers, alphas)
        train_error = np.mean(y_train_pred != y_train)
        training_errors.append(train_error)
        
        y_test_pred = predict(X_test, classifiers, alphas)
        test_error = np.mean(y_test_pred != y_test)
        test_errors.append(test_error)
        
        
        sample_weights *= np.exp(-alpha * np.array(y_train) * np.array(y_pred))
        sample_weights /= np.sum(sample_weights)
        
        
        stump_training_errors.append(epsilon)
        
        
        print(f"Iteration {t}: Ensemble Training Error = {train_error:.4f}, Ensemble Test Error = {test_error:.4f}")
        print(f"             Weighted Stump Training Error = {epsilon:.4f}")
    
    
    plot_errors(training_errors, test_errors, output_directory)
    plot_stump_errors(stump_training_errors, output_directory)
    return training_errors, test_errors, stump_training_errors

def plot_errors(training_errors, test_errors, output_directory):
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_errors) + 1), training_errors, label='Training Error')
    plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Errors vs. Number of Iterations')
    plt.legend()
    plt.tight_layout()
    
    
    os.makedirs(output_directory, exist_ok=True)
    
    output_path = os.path.join(output_directory, 'ensemble_error_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Ensemble error plot saved to {output_path}")

def plot_stump_errors(stump_training_errors, output_directory):
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(stump_training_errors) + 1), stump_training_errors, label='Weighted Stump Training Error')
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Error Rate')
    plt.title('Decision Stump Weighted Training Error vs. Number of Iterations')
    plt.legend()
    plt.tight_layout()
    
    
    os.makedirs(output_directory, exist_ok=True)
    
    output_path = os.path.join(output_directory, 'stump_error_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Decision stump error plot saved to {output_path}")
