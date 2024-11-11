import numpy as np

def voted_perceptron(X_train, y_train, T):
    """
    Voted Perceptron Algorithm with Bias Term.

    Parameters:
    - X_train: numpy array of shape (n_samples, n_features)
    - y_train: numpy array of shape (n_samples,)
    - T: Number of epochs

    Returns:
    - w_list: List of weight vectors
    - c_list: Corresponding counts for each weight vector
    """
    w = np.zeros(X_train.shape[1])
    w_list = []
    c_list = []
    c = 1  

    for epoch in range(T):
        for xi, yi in zip(X_train, y_train):
            if yi * np.dot(w, xi) <= 0:
                w_list.append(w.copy())
                c_list.append(c)
                w = w + yi * xi
                c = 1
            else:
                c += 1
    w_list.append(w.copy())
    c_list.append(c)
    return w_list, c_list
