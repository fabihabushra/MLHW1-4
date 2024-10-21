

import numpy as np
import matplotlib.pyplot as plt
import os

def batch_gradient_descent(X_train, y_train, X_test, y_test, initial_r, tolerance=1e-6, max_iterations=50000, decay_factor=0.8, decay_interval=500, results_directory=None):
    
    n_samples, n_features = X_train.shape
    n_test_samples = X_test.shape[0]
    w = np.zeros(n_features)  
    train_cost_history = []
    test_cost_history = []
    lr_history = []
    r = initial_r
    prev_cost = float('inf')

    for iteration in range(1, max_iterations + 1):
        
        y_pred_train = X_train.dot(w)
        
        error_train = y_pred_train - y_train
        
        cost_train = (1 / (2 * n_samples)) * np.sum(error_train ** 2)
        train_cost_history.append(cost_train)

        
        y_pred_test = X_test.dot(w)
        
        error_test = y_pred_test - y_test
        
        cost_test = (1 / (2 * n_test_samples)) * np.sum(error_test ** 2)
        test_cost_history.append(cost_test)

        
        lr_history.append(r)

        
        if iteration % decay_interval == 0:
            r *= decay_factor
            print(f"Iteration {iteration}: Decaying learning rate to {r}")

        
        if cost_train > prev_cost:
            
            continue

        prev_cost = cost_train

        
        gradient = (1 / n_samples) * X_train.T.dot(error_train)
        
        w_new = w - r * gradient

        
        weight_diff = np.linalg.norm(w_new - w)
        if weight_diff < tolerance:
            print(f"Converged after {iteration} iterations.")
            w = w_new
            break

        w = w_new

    else:
        print(f"Reached maximum iterations ({max_iterations}) without convergence.")

    
    if results_directory is not None:
        os.makedirs(results_directory, exist_ok=True)
        plot_cost_function(train_cost_history, test_cost_history,
                           'Cost Function vs. Iterations (Batch Gradient Descent)',
                           os.path.join(results_directory, 'cost_function_plot_batch.png'))
        plot_learning_rate(lr_history, 'Learning Rate vs. Iterations (Batch Gradient Descent)',
                           os.path.join(results_directory, 'learning_rate_plot_batch.png'))
        print(f"Plots saved to {results_directory}")

    return w

def plot_cost_function(train_cost_history, test_cost_history, title, output_path):
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_cost_history) + 1), train_cost_history, label='Training Cost')
    plt.plot(range(1, len(test_cost_history) + 1), test_cost_history, label='Test Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_learning_rate(lr_history, title, output_path):
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_history) + 1), lr_history)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
