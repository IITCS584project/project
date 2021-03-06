import numpy as np
class FactorLinearAnalyst:
    def __init__(self):
        pass

    def Fit(self, X, y):
        X = np.c_[X, np.zeros(len(y))]
        self.weights = np.zeros(len(y))
        pass