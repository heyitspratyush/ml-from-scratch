import numpy as np

def mean_squared_error(y,y_pred):
    if(y.shape != y_pred.shape):
        raise ValueError("Input shapes doesn't match") 
    y_diff = y-y_pred
    y_diff_squared = y_diff**2
    mse =  np.mean(y_diff_squared)
    return mse

