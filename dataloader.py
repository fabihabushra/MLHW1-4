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

def handle_unknown_values(data, categorical_indices):
    for col in categorical_indices:
        column_values = [row[col] for row in data if row[col] != 'unknown']
        
        if column_values:  
            most_common_value = Counter(column_values).most_common(1)[0][0]
        
            for row in data:
                if row[col] == 'unknown':
                    row[col] = most_common_value
                
    return data

def process_numerical_attributes(data, numerical_indices):
    data = np.array(data) 
    
    for index in numerical_indices:
        column = data[:, index]

        try:
            column = column.astype(float)
        except ValueError:
            print(f"Non-numeric values found in numerical attribute {index}: {set(column)}")
            continue

        median = np.median(column)
        binary_column = (column >= median).astype(int)
        data[:, index] = binary_column
    
    return data.tolist()

