import numpy as np
import os
import matplotlib.pyplot as plt

def train_svm_primal(X_train, y_train, X_test, y_test, C_values, epochs, schedule_type, results_dir):

    gamma_0 = 0.01
    a = 0.85

    def learning_rate(t, epoch, gamma_0, a, schedule_type):
        if schedule_type == 1:
            return gamma_0 / (1 + (gamma_0 / a) * t)
        elif schedule_type == 2:
            return gamma_0 / (1 + t)

    for C in C_values:
        print(f"Training with C = {C}")
        N, d = X_train.shape
        w = np.zeros(d)  
        train_errors = []
        test_errors = []

        updates = 0

        for epoch in range(epochs):
            indices = np.arange(N)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i, (x_i, y_i) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
                t = epoch * N + i
                gamma_t = learning_rate(t, epoch, gamma_0, a, schedule_type)

                margin = y_i * np.dot(w, x_i)

                if margin <= 1:
                    gradient = w - C * N * y_i * x_i
                else:
                    gradient = w

                w -= gamma_t * gradient

                # Error tracking
                train_predictions = np.sign(np.dot(X_train, w))
                test_predictions = np.sign(np.dot(X_test, w))
                train_error = np.mean(train_predictions != y_train)
                test_error = np.mean(test_predictions != y_test)
                train_errors.append(train_error)
                test_errors.append(test_error)

                updates += 1

        print(f"Final Training Error: {train_errors[-1]}")
        print(f"Final Test Error: {test_errors[-1]}")
        print(f"Learned Weights: {w}")

        # Save plots
        plt.figure(figsize=(10, 5))
        plt.plot(train_errors, label="Training Error", color="blue")
        plt.plot(test_errors, label="Test Error", color="orange")
        plt.title(f"Training and Test Error for C={C}")
        plt.xlabel("Number of Updates")
        plt.ylabel("Error")
        plt.legend()
        plot_file = os.path.join(results_dir, f"svm_primal_errors_C{C}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Error plot saved to {plot_file}\n")
