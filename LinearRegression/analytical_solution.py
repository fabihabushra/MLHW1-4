

import numpy as np

def analytical_solution(X_train, y_train):
    
    XtX = X_train.T.dot(X_train)
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_train.T.dot(y_train)
    w = XtX_inv.dot(XtY)
    return w
