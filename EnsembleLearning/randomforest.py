

import csv
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
import os
import random

def load_data(filename):
    
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def process_numerical_attributes(data, numerical_indices, medians=None):
    
    data = np.array(data)
    if medians is None:
        medians = {}
        compute_median = True
    else:
        compute_median = False

    for index in numerical_indices:
        column = data[:, index]

        
        column_numeric = []
        for val in column:
            try:
                column_numeric.append(float(val))
            except ValueError:
                column_numeric.append(np.nan)
        column_numeric = np.array(column_numeric)

        
        if compute_median:
            median = np.nanmedian(column_numeric)
            medians[index] = median
        else:
            median = medians[index]

        inds = np.where(np.isnan(column_numeric))
        column_numeric[inds] = median

        
        binary_column = (column_numeric >= median).astype(int)
        data[:, index] = binary_column
    return data.tolist(), medians

def one_hot_encode(data, categorical_indices):
    
    data_array = np.array(data)
    attribute_value_mapping = {}
    for index in categorical_indices:
        unique_values = np.unique(data_array[:, index])
        attribute_value_mapping[index] = list(unique_values)

    transformed_data = []
    for row in data_array:
        transformed_row = []
        for index in range(len(row)):
            if index in categorical_indices:
                
                value_list = attribute_value_mapping[index]
                one_hot = [0] * len(value_list)
                value = row[index]
                one_hot[value_list.index(value)] = 1
                transformed_row.extend(one_hot)
            else:
                
                transformed_row.append(int(row[index]))
        transformed_data.append(transformed_row)
    return transformed_data

def prepare_data(filename, numerical_indices, categorical_indices, medians=None):
    
    data = load_data(filename)
    
    data, medians = process_numerical_attributes(data, numerical_indices, medians)

    
    labels = [1 if row[-1] == 'yes' else -1 for row in data]

    
    data = [row[:-1] for row in data]

    
    data = one_hot_encode(data, categorical_indices)

    return data, labels, medians

class RandomForestTree:
    def __init__(self, max_features):
        self.tree = None
        self.max_features = max_features  

    def fit(self, X, y):
        feature_indices = list(range(len(X[0])))
        self.tree = self.build_tree(X, y, feature_indices)

    def predict(self, X):
        return [self.predict_sample(self.tree, sample) for sample in X]

    def predict_sample(self, node, sample):
        if node['is_leaf']:
            return node['label']
        feature_value = sample[node['feature_index']]
        if feature_value in node['children']:
            return self.predict_sample(node['children'][feature_value], sample)
        else:
            
            return node['majority_label']

    def build_tree(self, X, y, feature_indices):
        
        if len(set(y)) == 1:
            return {'is_leaf': True, 'label': y[0]}
        if not feature_indices:
            majority_label = Counter(y).most_common(1)[0][0]
            return {'is_leaf': True, 'label': majority_label}

        
        if len(feature_indices) <= self.max_features:
            features_to_consider = feature_indices.copy()
        else:
            features_to_consider = random.sample(feature_indices, self.max_features)

        
        best_feature = self.select_best_feature(X, y, features_to_consider)
        if best_feature is None:
            
            majority_label = Counter(y).most_common(1)[0][0]
            return {'is_leaf': True, 'label': majority_label}

        tree = {'is_leaf': False, 'feature_index': best_feature, 'children': {}, 'majority_label': Counter(y).most_common(1)[0][0]}

        
        feature_values = set([sample[best_feature] for sample in X])

        
        new_feature_indices = feature_indices.copy()
        new_feature_indices.remove(best_feature)

        for value in feature_values:
            
            indices = [i for i, sample in enumerate(X) if sample[best_feature] == value]
            X_subset = [X[i] for i in indices]
            y_subset = [y[i] for i in indices]

            if not X_subset:
                
                majority_label = Counter(y).most_common(1)[0][0]
                tree['children'][value] = {'is_leaf': True, 'label': majority_label}
            else:
                
                subtree = self.build_tree(X_subset, y_subset, new_feature_indices)
                tree['children'][value] = subtree

        return tree

    def select_best_feature(self, X, y, feature_indices):
        max_info_gain = -float('inf')
        best_feature = None
        current_entropy = self.calculate_entropy(y)

        for feature in feature_indices:
            
            feature_values = set([sample[feature] for sample in X])
            feature_entropy = 0

            for value in feature_values:
                
                indices = [i for i, sample in enumerate(X) if sample[feature] == value]
                y_subset = [y[i] for i in indices]
                weight = len(y_subset) / len(y)
                subset_entropy = self.calculate_entropy(y_subset)
                feature_entropy += weight * subset_entropy

            info_gain = current_entropy - feature_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature

        return best_feature

    def calculate_entropy(self, y):
        total = len(y)
        counts = Counter(y)
        entropy = 0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log2(probability)
        return entropy

def random_forest(X_train, y_train, X_test, y_test, num_trees, max_features):
    
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

        
        tree = RandomForestTree(max_features)
        tree.fit(X_bootstrap, y_bootstrap)
        trees.append(tree)

        
        y_train_pred = tree.predict(X_train)
        y_train_preds += y_train_pred

        
        y_test_pred = tree.predict(X_test)
        y_test_preds += y_test_pred

        
        y_train_majority = np.sign(y_train_preds)
        y_test_majority = np.sign(y_test_preds)

        
        train_error = np.mean(y_train_majority != y_train)
        test_error = np.mean(y_test_majority != y_test)

        training_errors.append(train_error)
        test_errors.append(test_error)

        
        if t % 10 == 0 or t == 1 or t == num_trees:
            print(f"Number of Trees: {t}, Max Features: {max_features}, Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    return training_errors, test_errors

def plot_random_forest_errors(errors_dict, output_directory):
    
    plt.figure(figsize=(12, 8))
    for max_features, (training_errors, test_errors) in errors_dict.items():
        plt.plot(range(1, len(test_errors) + 1), test_errors, label=f'Test Error (max_features={max_features})')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Test Errors vs. Number of Trees in Random Forest')
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_directory, 'random_forest_error_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"\nRandom Forest error plot saved to {output_path}")
