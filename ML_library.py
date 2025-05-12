import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import randint
from scipy.optimize import minimize

class GarchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self, 
            omega, 
            alpha, 
            beta
            ):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def garch_var(self, data:np.array, params:list):
        omega, alpha, beta = params
        var = np.zeros(len(data))
        var[0] = data[0] ** 2
        for i in range(1, len(data)):
            var[i] = omega + alpha * data[i-1] ** 2 + beta * var[i-1]
        return var
    
    def max_likelihood(self, params: list, data: np.array) -> float:
        cond_var = self.garch_var(data, params)
        return -np.sum(-np.log(cond_var) - data**2 / cond_var)

    def fit(self, X, y = None):
        omega, alpha, beta = self.omega, self.alpha, self.beta
        initial_guess = [omega, alpha, beta]
        ret = X['Return']
        res = minimize(self.max_likelihood, initial_guess, args = (ret,), method = 'nelder-mead', options={'disp': False})
        self.params = list(res.x)
        return self
    
    def predict(self, X):
        params = self.params
        ret = X['Return']
        var = self.garch_var(ret, params)
        return np.sqrt(12*var)
    
class EWMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam):
        self.lam = lam

    def ewma_var(self, data:np.array, param: float):
        lam = param
        var = np.zeros(len(data))
        var[0] = data[0] ** 2
        for i in range(1, len(data)):
            var[i] = (1-lam) * data[i-1] ** 2 + lam * var[i-1]
        return var
    
    def max_likelihood(self, params: list, data: np.array) -> float:
        cond_var = self.ewma_var(data, params)
        return -np.sum(-np.log(cond_var) - data**2 / cond_var)

    def fit(self, X, y = None):
        initial_guess = self.lam
        ret = X['Return']
        res = minimize(self.max_likelihood, initial_guess, args = (ret,), method = 'nelder-mead', options={'disp': False})
        self.param = res.x
        return self
    
    def predict(self, X):
        param = self.param
        ret = X['Return']
        var = self.ewma_var(ret, param)
        return np.sqrt(12*var)
    
class MaxGarchEWMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self, garch_model, ewma_model):
        self.garch_model = garch_model
        self.ewma_model = ewma_model

    def fit(self, X, y = None):
        self.garch_model.fit(X)
        self.ewma_model.fit(X)
        return self
    
    def predict(self, X):
        garch_pred = self.garch_model.predict(X)
        ewma_pred = self.ewma_model.predict(X)
        return np.maximum(garch_pred, ewma_pred)