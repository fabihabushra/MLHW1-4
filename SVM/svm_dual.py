import numpy as np
from scipy.optimize import minimize


def dual_objective(alpha, X, y):
    K = np.dot(X, X.T) 
    return -np.sum(alpha) + 0.5 * np.sum((y[:, None] * y[None, :]) * (alpha[:, None] * alpha[None, :]) * K)


def equality_constraint(alpha, y):
    return np.dot(alpha, y)


def train_svm_dual(X, y, C):

    n_samples, n_features = X.shape
    alpha0 = np.zeros(n_samples)
    bounds = [(0, C) for _ in range(n_samples)]
    
    constraints = {
        'type': 'eq',
        'fun': equality_constraint,
        'args': (y,)
    }
    
    result = minimize(
        dual_objective,
        alpha0,
        args=(X, y),
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"disp": True}
    )
    
    alpha = result.x
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    support_vectors = (alpha > 1e-6) & (alpha < C - 1e-6)
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    
    return w, b, alpha, support_vectors


def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)
