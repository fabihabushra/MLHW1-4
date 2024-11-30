import csv
import numpy as np
from collections import Counter

def load_data(filename):
    
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data

def load_data_concrete(filename):
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                row = line.split(',')
            else:
                row = line.split()
            
            row = [float(value) for value in row]
            data.append(row)
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def add_bias_term(X):
    
    return np.hstack((np.ones((X.shape[0], 1)), X))

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

def handle_unknown_values(data, categorical_indices):
    
    data = np.array(data)
    for index in categorical_indices:
        
        column = data[:, index]
        
        non_unknown = column[column != 'unknown']
        if len(non_unknown) == 0:
            
            continue
        most_common = Counter(non_unknown).most_common(1)[0][0]
        
        column[column == 'unknown'] = most_common
        data[:, index] = column
    return data.tolist()

def process_numerical_attributes(data, numerical_indices, medians=None):
    
    data = np.array(data, dtype=object)  
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
                if value in value_list:
                    one_hot[value_list.index(value)] = 1
                transformed_row.extend(one_hot)
            else:
                
                transformed_row.append(int(row[index]))
        transformed_data.append(transformed_row)
    return transformed_data

def prepare_data_bias_variance(filename, numerical_indices, categorical_indices, medians=None):
    
    data = load_data(filename)
    
    data, medians = process_numerical_attributes(data, numerical_indices, medians)

    
    labels = [1 if row[-1] == 'yes' else -1 for row in data]

    
    data = [row[:-1] for row in data]

    
    data = one_hot_encode(data, categorical_indices)

    return data, labels, medians

def load_data_with_bias(filename):

    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y == 0, -1, 1)
    X = np.hstack((X, np.ones((X.shape[0], 1))))  
    return X, y

def load_data_bank_note(filename):

    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    y = np.where(y == 0, -1, 1)  
    return X, y
