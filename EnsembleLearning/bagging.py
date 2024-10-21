

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os


from DecisionTree.decisiontree import DecisionTree
from dataloader import load_data, process_numerical_attributes, one_hot_encode

def prepare_data(filename, numerical_indices, categorical_indices, medians=None):
    
    data = load_data(filename)
    data, medians = process_numerical_attributes(data, numerical_indices, medians)
    labels = [1 if row[-1].lower() == 'yes' else -1 for row in data]
    data = [row[:-1] for row in data]
    data = one_hot_encode(data, categorical_indices)
    return data, labels, medians

def bagging(X_train, y_train, X_test, y_test, num_trees):
    
    n_samples = len(X_train)
    trees = []
    training_errors = []
    test_errors = []
    
    
    y_train_preds = np.zeros(n_samples)
    y_test_preds = np.zeros(len(X_test))
    
    for t in range(1, num_trees + 1):
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = [X_train[i] for i in indices]
        y_bootstrap = [y_train[i] for i in indices]
        
        
        tree = DecisionTree()
        tree.fit(X_bootstrap, y_bootstrap)
        trees.append(tree)
        
        
        y_train_pred = tree.predict(X_train)
        y_train_preds += y_train_pred
        
        
        y_test_pred = tree.predict(X_test)
        y_test_preds += y_test_pred
        
        
        y_train_majority = np.sign(y_train_preds)
        y_test_majority = np.sign(y_test_preds)
        
        
        y_train_majority[y_train_majority == 0] = 1
        y_test_majority[y_test_majority == 0] = 1
        
        
        train_error = np.mean(y_train_majority != y_train)
        test_error = np.mean(y_test_majority != y_test)
        
        training_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f"Number of Trees: {t}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    return training_errors, test_errors

def bagging_predictor(X_train, y_train, num_trees):
    
    n_samples = len(X_train)
    trees = []

    for t in range(num_trees):
        
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = [X_train[i] for i in indices]
        y_bootstrap = [y_train[i] for i in indices]

        
        tree = DecisionTree()
        tree.fit(X_bootstrap, y_bootstrap)
        trees.append(tree)

    return trees


def plot_bagging_errors(training_errors, test_errors, output_directory):
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_errors) + 1), training_errors, label='Training Error')
    plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Errors vs. Number of Trees in Bagging')
    plt.legend()
    plt.tight_layout()
    
    
    os.makedirs(output_directory, exist_ok=True)
    
    output_path = os.path.join(output_directory, 'bagging_error_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Bagging error plot saved to {output_path}")
