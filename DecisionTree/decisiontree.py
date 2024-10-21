# DecisionTree/decisiontree.py

from collections import Counter
import math
import numpy as np

def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(labels)
    entropy_value = 0
    for count in label_counts.values():
        p = count / total
        entropy_value -= p * math.log2(p) if p > 0 else 0  # avoid log(0)
    return entropy_value

def info_gain(data, attribute_index, numerical_indices=None):
    total_entropy = entropy(data)
    values = [row[attribute_index] for row in data]
    value_counts = Counter(values)
    total = len(values)
    weighted_entropy = 0
    for value, count in value_counts.items():
        subset = [row for row in data if row[attribute_index] == value]
        weighted_entropy += (count / total) * entropy(subset)
    return total_entropy - weighted_entropy

def gini_index(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(labels)
    gini = 1
    for count in label_counts.values():
        p = count / total
        gini -= p ** 2
    return gini

def gini_gain(data, attribute_index, numerical_indices=None):
    total_gini = gini_index(data)
    values = [row[attribute_index] for row in data]
    value_counts = Counter(values)
    total = len(values)
    weighted_gini = 0
    for value, count in value_counts.items():
        subset = [row for row in data if row[attribute_index] == value]
        weighted_gini += (count / total) * gini_index(subset)
    return total_gini - weighted_gini

def majority_error(data):
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    total = len(labels)
    majority_count = max(label_counts.values())
    return 1 - (majority_count / total)

def majority_error_gain(data, attribute_index, numerical_indices=None):
    total_error = majority_error(data)
    values = [row[attribute_index] for row in data]
    value_counts = Counter(values)
    total = len(values)
    weighted_error = 0
    for value, count in value_counts.items():
        subset = [row for row in data if row[attribute_index] == value]
        weighted_error += (count / total) * majority_error(subset)
    return total_error - weighted_error

def id3(data, attributes, depth, max_depth, criterion='information_gain', numerical_indices=None):
    labels = [row[-1] for row in data]
    
    # Pure node
    if len(set(labels)) == 1:
        return labels[0]
    
    # Max depth reached or no more attributes
    if not attributes or depth == max_depth:
        return Counter(labels).most_common(1)[0][0]
    
    # Calculate gains based on the criterion
    if criterion == 'information_gain':
        gains = [info_gain(data, i, numerical_indices) for i in attributes]
    elif criterion == 'gini':
        gains = [gini_gain(data, i, numerical_indices) for i in attributes]
    elif criterion == 'majority_error':
        gains = [majority_error_gain(data, i, numerical_indices) for i in attributes]
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    # Select the best attribute to split on
    best_attr = attributes[gains.index(max(gains))]

    tree = {best_attr: {}}
    
    if numerical_indices and best_attr in numerical_indices:
        # For numerical attributes, split based on median
        median = np.median([float(row[best_attr]) for row in data])
        
        greater_equal_subset = [row for row in data if float(row[best_attr]) >= median]
        less_subset = [row for row in data if float(row[best_attr]) < median]

        if greater_equal_subset:
            tree[best_attr][f">= {median}"] = id3(greater_equal_subset, attributes, depth + 1, max_depth, criterion, numerical_indices)
        if less_subset:
            tree[best_attr][f"< {median}"] = id3(less_subset, attributes, depth + 1, max_depth, criterion, numerical_indices)
    
    else:
        # For categorical attributes, split based on each unique value
        values = set([row[best_attr] for row in data])
        for value in values:
            subset = [row for row in data if row[best_attr] == value]
            if subset:
                tree[best_attr][value] = id3(subset, [a for a in attributes if a != best_attr], depth + 1, max_depth, criterion, numerical_indices)
            else:
                tree[best_attr][value] = Counter(labels).most_common(1)[0][0]

    return tree

def classify(tree, instance):
    """
    Classify a single instance using the decision tree.
    """
    if not isinstance(tree, dict):
        return tree  # Leaf node
    
    attr = next(iter(tree))  
    value = instance[attr] 
    
    if isinstance(list(tree[attr].keys())[0], str) and '>=' in list(tree[attr].keys())[0]:
        for threshold in tree[attr]:
            condition, threshold_value = threshold.split(' ')
            threshold_value = float(threshold_value)
            
            if condition == '>=' and float(value) >= threshold_value:
                return classify(tree[attr][threshold], instance)
            elif condition == '<' and float(value) < threshold_value:
                return classify(tree[attr][threshold], instance)
    else:
        if value in tree[attr]:
            return classify(tree[attr][value], instance)
        else:
            # If the value was not seen during training, return the majority label
            labels = []
            for subtree in tree[attr].values():
                if isinstance(subtree, dict):
                    # If subtree is another node, collect its majority label
                    # This requires a helper function or additional data structure
                    # For simplicity, return 1 or -1
                    labels.append(1)
                else:
                    labels.append(subtree)
            return 1 if sum(labels) >= 0 else -1 

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y, feature_indices=None):
        if feature_indices is None:
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
            # Handle unseen feature values by using majority label
            return node['majority_label']

    def build_tree(self, X, y, feature_indices):
        # Base cases
        if len(set(y)) == 1:
            return {'is_leaf': True, 'label': y[0]}
        if not feature_indices:
            majority_label = Counter(y).most_common(1)[0][0]
            return {'is_leaf': True, 'label': majority_label}

        # Select the best feature to split
        best_feature = self.select_best_feature(X, y, feature_indices)
        tree = {
            'is_leaf': False,
            'feature_index': best_feature,
            'children': {},
            'majority_label': Counter(y).most_common(1)[0][0]
        }

        # Get unique values of the best feature
        feature_values = set([sample[best_feature] for sample in X])

        # Remove the best feature from the list
        new_feature_indices = feature_indices.copy()
        new_feature_indices.remove(best_feature)

        for value in feature_values:
            # Split the dataset
            indices = [i for i, sample in enumerate(X) if sample[best_feature] == value]
            X_subset = [X[i] for i in indices]
            y_subset = [y[i] for i in indices]

            if not X_subset:
                # Create a leaf node with the majority label
                majority_label = Counter(y).most_common(1)[0][0]
                tree['children'][value] = {'is_leaf': True, 'label': majority_label}
            else:
                # Recursively build the tree
                subtree = self.build_tree(X_subset, y_subset, new_feature_indices)
                tree['children'][value] = subtree

        return tree

    def select_best_feature(self, X, y, feature_indices):
        max_info_gain = -float('inf')
        best_feature = None
        current_entropy = self.calculate_entropy(y)

        for feature in feature_indices:
            # Get all possible values of the feature
            feature_values = set([sample[feature] for sample in X])
            feature_entropy = 0

            for value in feature_values:
                # Subset the data for each value
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
            entropy -= probability * math.log2(probability) if probability > 0 else 0
        return entropy
