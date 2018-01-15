from sklearn.linear_model import LogisticRegression
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
        
    def dX(self, X, Y):
        h = (X*self.w).sum(axis=1) + self.b
        Yhat = logistic.cdf(h)
        dif = Yhat-Y
        sigma = dif * Yhat * (1-Yhat)
        dx = (sigma.reshape(-1,1)* self.w)
    
    def predict(self, X):
        h = (X*self.w).sum(axis=1) + self.b
        Yhat = logistic.cdf(h)
        return (Yhat>0.5)+0


class SkClassifier(object):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.raw_cls = LogisticRegression(**params)

    def fit(self, X, Y, epoch=1000):
        self.raw_cls.fit(X, Y)
            
    def dx(self, X, Y):
        b = self.raw_cls.intercept_[0]
        theta = self.raw_cls.coef_[0]

        h = (X*theta).sum(axis=1) + b
        Yhat = logistic.cdf(h)
        dif = Yhat-Y
        sigma = dif * Yhat * (1-Yhat)
        dx = (sigma.reshape(-1,1)* theta)
        return dx

    
    def predict(self, X):
        return self.raw_cls.predict(X)
