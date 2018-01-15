from scipy.stats import logistic
import numpy as np

class Classifier(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, X, Y, epoch=1000):
        self.w = np.random.random(X.shape[1])*0.1 - 0.05
        self.b = np.random.random()*0.1 - 0.05
        for e in range(epoch):
            self.update(X, Y)
    
    def update(self, X, Y):
        h = (X*self.w).sum(axis=1) + self.b
        Yhat = logistic.cdf(h)
        dif = Yhat-Y
        sigma = dif * Yhat * (1-Yhat)
        dw = (sigma.reshape(-1,1)* X).sum(axis=0)
        db = sigma.sum()
        self.b -= db
        self.w -= dw
        
    def gradient(X, Y):
        pass
    
    def predict(self, X):
        h = (X*self.w).sum(axis=1) + self.b
        Yhat = logistic.cdf(h)
        return (Yhat>0.5)+0