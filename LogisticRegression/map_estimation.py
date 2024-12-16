import numpy as np
from ..dataloader import load_data_nn_nn

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def neg_log_posterior(w, X, y, v):
    N = X.shape[0]
    z = X @ w
    eps = 1e-15
    pred = sigmoid(z)
    ll = np.sum(y*np.log(pred+eps) + (1-y)*np.log(1-pred+eps))
    prior = -(np.sum(w**2)/(2*v))
    return -(ll + prior)

def grad_neg_log_posterior_single(w, x, y, v):
    pred = sigmoid(np.dot(w, x))
    return -(y - pred)*x + (w / v)

def compute_error(w, X, y):
    pred = sigmoid(X @ w) >= 0.5
    return np.mean(pred != y)

def run_map_logistic_regression(train_file, test_file, variances, gamma0=0.1, d=100.0, T=100):
    X_train, y_train = load_data_nn(train_file)
    X_test, y_test = load_data_nn(test_file)

    mean_X = X_train.mean(axis=0)
    std_X = X_train.std(axis=0)
    std_X[std_X == 0] = 1.0
    X_train = (X_train - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    X_train = np.hstack([np.ones((N_train,1)), X_train])
    X_test = np.hstack([np.ones((N_test,1)), X_test])

    results = {}
    for v in variances:
        w = np.zeros(X_train.shape[1])
        t = 0
        
        for epoch in range(T):
            perm = np.random.permutation(N_train)
            X_train = X_train[perm]
            y_train = y_train[perm]
            
            for i in range(N_train):
                t += 1
                gamma_t = gamma0 / (1 + (gamma0/d)*t)
                g = grad_neg_log_posterior_single(w, X_train[i], y_train[i], v)
                w = w - gamma_t * g

        train_err = compute_error(w, X_train, y_train)
        test_err = compute_error(w, X_test, y_test)
        results[v] = (train_err, test_err)

    return results
