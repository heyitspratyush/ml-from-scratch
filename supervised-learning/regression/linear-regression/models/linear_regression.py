
import numpy as np
from src.gradients import calculate_gradients
from src.predict import predict as training_predict
from src.update import update_parameters 
class LinearRegression:
    def __init__(self,learning_rate,n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self,X,y):
        self.w = np.zeros((X.shape[1],1))
        self.b = 0.0
        for _ in range(self.n_iterations):
            y_pred = training_predict(X,self.w,self.b)
            dw,db = calculate_gradients(X,y,y_pred)
            w_new,b_new = update_parameters(self.w,self.b,dw,db,self.learning_rate)
            self.w,self.b =w_new,b_new

        return self

    def predict(self,X_new):
        if self.w is None or self.b is None:
            raise RuntimeError("Model has not been trained yet")
        y_new = training_predict(X_new,self.w,self.b)
        return y_new 
            








