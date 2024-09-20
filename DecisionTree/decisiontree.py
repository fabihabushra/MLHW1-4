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
    
    # for pure node
    if len(set(labels)) == 1:
        return labels[0]
    
    if not attributes or depth == max_depth:
        return Counter(labels).most_common(1)[0][0]
    
    # # Debug
    # for attr in attributes:
    #     unique_values = Counter([row[attr] for row in data])
    #     print(f"Attribute {attr} unique values: {unique_values}")

    if criterion == 'information_gain':
        gains = [info_gain(data, i, numerical_indices) for i in attributes]
        
        # # Debug
        # for i, gain in zip(attributes, gains):
        #     print(f"Attribute {i} - Information Gain: {gain}")
        
    elif criterion == 'gini':
        gains = [gini_gain(data, i, numerical_indices) for i in attributes]
    elif criterion == 'majority_error':
        gains = [majority_error_gain(data, i, numerical_indices) for i in attributes]

    best_attr = attributes[gains.index(max(gains))]

    # # Debug
    # if criterion == 'information_gain':
    #     print(f"Depth {depth}: Best attribute to split on: {best_attr}, with Information Gain: {max(gains)}")

    tree = {best_attr: {}}
    
    if numerical_indices and best_attr in numerical_indices:
        median = np.median([float(row[best_attr]) for row in data])
        # # Debug
        # if criterion == 'information_gain':
        #     print(f"Splitting on numerical attribute {best_attr} with median {median}")
        
        greater_equal_subset = [row for row in data if float(row[best_attr]) >= median]
        less_subset = [row for row in data if float(row[best_attr]) < median]

        # # Debug
        # print(f"Data subset where {best_attr} >= {median}: {greater_equal_subset}")
        # print(f"Data subset where {best_attr} < {median}: {less_subset}")

        if greater_equal_subset:
            tree[best_attr][f">= {median}"] = id3(greater_equal_subset, attributes, depth + 1, max_depth, criterion, numerical_indices)
        if less_subset:
            tree[best_attr][f"< {median}"] = id3(less_subset, attributes, depth + 1, max_depth, criterion, numerical_indices)
    
    else:
        values = set([row[best_attr] for row in data])
        for value in values:
            subset = [row for row in data if row[best_attr] == value]
            if subset:
                # # Debug
                # if criterion == 'information_gain':
                #         print(f"Splitting on categorical attribute {best_attr} with value {value}")
                #         print(f"Data subset for {best_attr} = {value}: {subset}")
                tree[best_attr][value] = id3(subset, [a for a in attributes if a != best_attr], depth + 1, max_depth, criterion, numerical_indices)
            else:
                tree[best_attr][value] = Counter(labels).most_common(1)[0][0]

    return tree

def classify(tree, instance):
    if not isinstance(tree, dict):
        return tree  # leaf node
    
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
            # # Debug
            # print(f"Warning: No valid branch for attribute {attr} with value {value}")
            return None 

