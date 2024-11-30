import numpy as np
from scipy.optimize import minimize

def gaussian_kernel(x_i, x_j, gamma):
    return np.exp(-np.linalg.norm(x_i - x_j)**2 / gamma)


def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K


def dual_objective(alpha, K, y):
    return -np.sum(alpha) + 0.5 * np.sum((y[:, None] * y[None, :]) * (alpha[:, None] * alpha[None, :]) * K)


def equality_constraint(alpha, y):
    return np.dot(alpha, y)


def train_svm_dual_gaussian(X, y, C, gamma):
    n_samples = X.shape[0]
    K = compute_kernel_matrix(X, gamma)
    alpha0 = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    constraints = {'type': 'eq', 'fun': equality_constraint, 'args': (y,)}

    result = minimize(
        dual_objective,
        alpha0,
        args=(K, y),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"disp": False}
    )

    alpha = result.x
    support_vectors = (alpha > 1e-6) & (alpha < C - 1e-6)
    return alpha, support_vectors


def predict_gaussian(X_train, y_train, X_test, alpha, gamma):
    n_samples = X_train.shape[0]
    n_test_samples = X_test.shape[0]
    y_pred = np.zeros(n_test_samples)

    for i in range(n_test_samples):
        summation = 0
        for j in range(n_samples):
            summation += alpha[j] * y_train[j] * gaussian_kernel(X_train[j], X_test[i], gamma)
        y_pred[i] = np.sign(summation)

    return y_pred


def train_nonlinear_svm(X_train, y_train, X_test, y_test, C_values, gamma_values, results_dir):
    results = []
    support_vector_counts = {}
    support_vectors_all = {}

    for C in C_values:
        support_vector_counts[C] = {}
        support_vectors_all[C] = {}
        for gamma in gamma_values:
            print(f"Training with C = {C}, gamma = {gamma}")

            alpha, support_vectors = train_svm_dual_gaussian(X_train, y_train, C, gamma)
            train_predictions = predict_gaussian(X_train, y_train, X_train, alpha, gamma)
            test_predictions = predict_gaussian(X_train, y_train, X_test, alpha, gamma)

            train_error = np.mean(train_predictions != y_train)
            test_error = np.mean(test_predictions != y_test)

            support_vector_counts[C][gamma] = np.sum(support_vectors)
            support_vectors_all[C][gamma] = np.where(support_vectors)[0]
            results.append((C, gamma, train_error, test_error, np.sum(support_vectors)))

            print(f"Training Error: {train_error}, Test Error: {test_error}")
            print(f"Number of Support Vectors: {support_vector_counts[C][gamma]}")

    C_target = 500 / len(y_train)
    print(f"\nSupport Vector Overlaps for C = {C_target}:")
    for i in range(len(gamma_values) - 1):
        gamma1 = gamma_values[i]
        gamma2 = gamma_values[i + 1]
        sv1 = set(support_vectors_all[C_target][gamma1])
        sv2 = set(support_vectors_all[C_target][gamma2])
        overlap = len(sv1.intersection(sv2))
        print(f"Overlap between gamma = {gamma1} and gamma = {gamma2}: {overlap}")

    print("\nFinal Results:")
    print("C, Gamma, Training Error, Test Error, Number of Support Vectors")
    for result in results:
        print(result)
