import numpy as np

def standard_perceptron(X_train, y_train, T):
    """
    Standard Perceptron Algorithm with Bias Term.

    Parameters:
    - X_train: numpy array of shape (n_samples, n_features)
    - y_train: numpy array of shape (n_samples,)
    - T: Number of epochs

    Returns:
    - w: Learned weight vector
    """
    w = np.zeros(X_train.shape[1])

    for epoch in range(T):
        for xi, yi in zip(X_train, y_train):
            if yi * np.dot(w, xi) <= 0:
                w += yi * xi
    return w
