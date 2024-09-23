import os
from dataloader import load_data, handle_unknown_values, process_numerical_attributes
from DecisionTree.decisiontree import id3, classify

def calculate_error(tree, data, dataset_type="train"):
    correct = 0
    # print(f"\n--- {dataset_type.capitalize()} Predictions (First 5) ---")
    
    for i, row in enumerate(data):
        prediction = classify(tree, row)
        actual_label = row[-1] 
        
        # if i < 5:
        #     print(f"{dataset_type.capitalize()} Instance {i + 1}: Actual label = {actual_label}, Prediction = {prediction}")
        
        if prediction == actual_label:
            correct += 1
    
    accuracy = correct / len(data)
    # print(f"Correct predictions in {dataset_type}: {correct} / {len(data)}")
    return 1 - accuracy  

def infer_column_types(data, num_check_threshold=5):
    categorical_indices = []
    numerical_indices = []
    
    num_cols = len(data[0])
    for col_idx in range(num_cols):
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

def run_experiments_bank(train_file, test_file, max_depth):
    """Run the ID3 experiment on the bank dataset (with numerical attributes)."""
    train_data = load_data(train_file)
    test_data = load_data(test_file)
    
    categorical_indices, numerical_indices = infer_column_types(train_data)
    # print("Categorical Indices: ", categorical_indices)
    # print("Numerical Indices: ", numerical_indices)
    
    train_data = handle_unknown_values(train_data, categorical_indices)
    test_data = handle_unknown_values(test_data, categorical_indices)
    
    train_data = process_numerical_attributes(train_data, numerical_indices)
    test_data = process_numerical_attributes(test_data, numerical_indices)

    # print("Train Dataset after handling 'unknown' values:")
    # for row in train_data:
    #     print(row)
    
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

if __name__ == "__main__":
    print("Please select a dataset to run the experiment on:")
    print("1. Car Dataset")
    print("2. Bank Dataset")
    
    dataset_choice = input("Enter 1 or 2: ").strip()
    
    if dataset_choice == "1":
        # experiment on the car dataset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(base_dir, 'car', 'train.csv')
        test_file = os.path.join(base_dir, 'car', 'test.csv')
        max_depth = int(input("Enter the maximum depth of the tree (e.g. 6): "))
        run_experiments_car(train_file, test_file, max_depth)
        
    elif dataset_choice == "2":
        # experiment on the bank dataset
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_file = os.path.join(base_dir, 'bank', 'train.csv')
        test_file = os.path.join(base_dir, 'bank', 'test.csv')
        max_depth = int(input("Enter the maximum depth of the tree (e.g. 16): "))
        run_experiments_bank(train_file, test_file, max_depth)
        
    else:
        print("Invalid choice. Please enter 1 or 2.")
