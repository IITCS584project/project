import numpy as np
from sklearn.naive_bayes import GaussianNB
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
import torch
class GaussianNaiveBayesDemo:
    def __init__(self):
        self.mClf = None
        pass
    
    def Fit(self, X, y):
        self.mClf = GaussianNB()
        self.mClf.fit(X,y)

    def Predict(self, X):
        pred_y = self.mClf.predict(X)
        return pred_y

    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        return (pred_y == true_y).sum() / len(true_y)

def GenSamples(feature_cnt, sample_cnt):
    # feature_cnt-dimension vector, if the sample is in the sphere with radius 1, then it is class 1, else class 2
    X = np.random.rand(sample_cnt, feature_cnt)
    y = np.zeros(shape=(sample_cnt, 1))
    for k in range(sample_cnt):
        y[k] = np.dot( X[k], X[k] ) <= 1
    #y = X.mean(axis=1)
    #y[y>=0.5] = 1
    #y[y<0.5] = 0    
    return torch.tensor(X).float(), torch.tensor(y).float()

def Split(X, y):
    row_cnt = X.shape[0]
    split_point = int(row_cnt * 2 / 3)
    X_train = X[:split_point]
    X_test = X[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]
    return X_train, y_train, X_test, y_test

def Accuracy(pred, true_y):
    data_num :int = len(true_y)
    true_y = true_y.reshape((data_num,))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    correct_rate = (pred == true_y.numpy()).sum() / len(true_y)    
    print(correct_rate)

def Main():
    solver = GaussianNaiveBayesDemo()
    X,y = GenSamples(3, 500)
    X_train, y_train, X_test, y_test = Split(X,y)
    solver.Fit(X_train, y_train)
    pred_y = solver.Predict(X_test)
    Accuracy(pred_y, y_test)
    pass

Main()