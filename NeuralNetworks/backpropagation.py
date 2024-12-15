import numpy as np

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))

def binary_cross_entropy_loss(y, Y):
    epsilon = 1e-15
    Y_clipped = np.clip(Y, epsilon, 1 - epsilon)
    return -np.mean(y*np.log(Y_clipped) + (1 - y)*np.log(1 - Y_clipped))

def forward_pass(X, W1, W2, W3):
    N = X.shape[0]
    Z0_1 = np.ones((N, 1))
    A1_in = np.concatenate([Z0_1, X], axis=1)
    U1 = A1_in @ W1
    Z1 = sigmoid(U1)
    
    Z0_2 = np.ones((N, 1))
    A2_in = np.concatenate([Z0_2, Z1], axis=1)
    U2 = A2_in @ W2
    Z2 = sigmoid(U2)
    
    Z0_3 = np.ones((N,1))
    A3_in = np.concatenate([Z0_3, Z2], axis=1)
    U3 = A3_in @ W3
    Y = sigmoid(U3)
    return A1_in, Z1, A2_in, Z2, A3_in, Y, U1, U2, U3

def backprop(X, y, W1, W2, W3):
    A1_in, Z1, A2_in, Z2, A3_in, Y, U1, U2, U3 = forward_pass(X, W1, W2, W3)
    y = y.reshape(-1, 1)
    
    dL_dU3 = Y - y
    dL_dW3 = A3_in.T @ dL_dU3
    
    dL_dZ2 = dL_dU3 @ W3[1:].T
    dZ2_dU2 = Z2 * (1 - Z2)
    dL_dU2 = dL_dZ2 * dZ2_dU2
    dL_dW2 = A2_in.T @ dL_dU2
    
    dL_dZ1 = dL_dU2 @ W2[1:].T
    dZ1_dU1 = Z1 * (1 - Z1)
    dL_dU1 = dL_dZ1 * dZ1_dU1
    dL_dW1 = A1_in.T @ dL_dU1
    
    curr_loss = binary_cross_entropy_loss(y, Y)
    return dL_dW1, dL_dW2, dL_dW3, curr_loss

def compute_error(X, y, W1, W2, W3):
    _, _, _, _, _, Y, _, _, _ = forward_pass(X, W1, W2, W3)
    return np.mean((Y.flatten() >= 0.5).astype(int) != y)

def train_nn(X_train, y_train, X_test, y_test, hidden_width=10, gamma0=0.1, d=100.0, epochs=20):
    np.random.seed(42)
    N, d_in = X_train.shape
    
    W1 = np.random.randn(d_in+1, hidden_width)
    W2 = np.random.randn(hidden_width+1, hidden_width)
    W3 = np.random.randn(hidden_width+1, 1)
    
    t = 0
    epoch_losses = []
    
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        epoch_loss_accum = 0.0
        for i in range(N):
            x_i = X_train[i:i+1]
            y_i = y_train[i:i+1]
            
            t += 1
            gamma_t = gamma0 / (1 + (gamma0/d)*t)
            
            dL_dW1, dL_dW2, dL_dW3, curr_loss = backprop(x_i, y_i, W1, W2, W3)
            
            W1 -= gamma_t * dL_dW1
            W2 -= gamma_t * dL_dW2
            W3 -= gamma_t * dL_dW3
            
            epoch_loss_accum += curr_loss
        
        avg_epoch_loss = epoch_loss_accum / N
        epoch_losses.append(avg_epoch_loss)
        
    
    return W1, W2, W3, epoch_losses

def run_neural_network(X_train, y_train, X_test, y_test, widths, gamma0=0.1, d_val=100.0, epochs=20):
    results = {}
    epoch_losses_dict = {}
    
    for w in widths:
        W1, W2, W3, epoch_losses = train_nn(X_train, y_train, X_test, y_test,
                                            hidden_width=w, gamma0=gamma0, d=d_val, epochs=epochs)
        
        train_err = compute_error(X_train, y_train, W1, W2, W3)
        test_err = compute_error(X_test, y_test, W1, W2, W3)
        results[w] = (train_err, test_err)
        epoch_losses_dict[w] = epoch_losses
    
    return results, epoch_losses_dict
