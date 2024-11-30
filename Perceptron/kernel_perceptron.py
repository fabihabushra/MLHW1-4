import numpy as np


def gaussian_kernel(x_i, x_j, gamma):
    return np.exp(-np.linalg.norm(x_i - x_j) ** 2 / gamma)


def kernel_perceptron(X_train, y_train, X_test, y_test, gamma, max_iter=100):
    n_samples = X_train.shape[0]
    alphas = np.zeros(n_samples)  
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = gaussian_kernel(X_train[i], X_train[j], gamma)

    for _ in range(max_iter):
        for i in range(n_samples):
            prediction = np.sign(np.sum(alphas * y_train * kernel_matrix[:, i]))
            if prediction != y_train[i]: 
                alphas[i] += 1

    def predict(X_train, y_train, X_test, alphas, gamma):
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            summation = 0
            for j in range(X_train.shape[0]):
                summation += alphas[j] * y_train[j] * gaussian_kernel(X_train[j], X_test[i], gamma)
            y_pred[i] = np.sign(summation)
        return y_pred

    train_predictions = predict(X_train, y_train, X_train, alphas, gamma)
    test_predictions = predict(X_train, y_train, X_test, alphas, gamma)

    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    return train_error, test_error


def train_kernel_perceptron(X_train, y_train, X_test, y_test, gamma_values):
    results = []

    for gamma in gamma_values:
        print(f"Running Kernel Perceptron with gamma = {gamma}")
        train_error, test_error = kernel_perceptron(X_train, y_train, X_test, y_test, gamma)
        results.append((gamma, train_error, test_error))
        print(f"Training Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    return results
