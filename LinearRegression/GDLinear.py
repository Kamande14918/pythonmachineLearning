# Implementation of gradient descent in linear regression
import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression:
    def __init__(self, X ,Y):
        self.X = X
        self.Y = Y
        self.b = [0,0]
        
        
    def update_coeffs(self, learning_rate):
        Y_pred = self.predict()
        Y = self.Y
        m = len(Y)
        
        self.b[0] = self.b[0] - (learning_rate* 
                                 ((1/m)* np.sum(Y_pred -Y)))
        self.b[1] = self.b[1] - (learning_rate *
                                 ((1/m) * np.sum((Y_pred - Y) * self.X)))
        def predict(self,X=[]):
            Y_pred = np.array([])
            if not X:
                X = self.X
            b = self.b
            for x in X:
                Y_pred = np.append(Y_pred, b[0] + (b[1] * x))
                return Y_pred
            
            
            def get_current_accuracy(self,Y_pred):
                p ,e = Y_pred, self.Y
                n = len(Y_pred)
                return 1 - sum(
                   
                )