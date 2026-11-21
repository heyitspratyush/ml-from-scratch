
import numpy as np


def calculate_gradients(X, y, y_pred):

    # X must be a 2D feature matrix
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    # y must be a 2D column vector
    if y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y must be a 2D column vector")

    # Predictions must also be a 2D column vector
    if y_pred.ndim != 2 or y_pred.shape[1] != 1:
        raise ValueError("y_pred must be a 2D column vector")

    # Actual values and predictions must have identical shapes
    if y.shape != y_pred.shape:
        raise ValueError("Input shapes do not match")

    # Number of observations must match
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of observations does not match")

    # Calculate residuals
    residuals = y_pred - y

    # Calculate gradients for weights and bias
    dw = (2 / X.shape[0]) * (X.T @ residuals)
    db = 2 * np.mean(residuals)

    return dw, db
