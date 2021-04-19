import numpy as np
from sklearn.naive_bayes import GaussianNB
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcPredictResult
from Data.UseData import read_data
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.Common.StockDataProvider import StockDataProvider
class GaussianNaiveBayesMethod:
    def __init__(self):
        self.mClf = None
        pass
    
    def LoadData(self, ticker_list, start_date, end_date, distance, columns):
        reader = read_data()
        stock_data = reader.get_daily_data(ticker_list,start_date, end_date, distance,columns)
        # the first row of yield is na
        stock_data = stock_data[:, 1:]
        return stock_data
    
    def Fit(self, X, y):
        self.mClf = GaussianNB()
        #X = X.reshape((len(X),1))
        self.mClf.fit(X,y)

    def Predict(self, X):
        #X = X.reshape((len(X),1))
        pred_y = self.mClf.predict(X)
        return pred_y

    def Accuracy(self, pred_y, true_y):
        '''
        calculate the accuracy of the prediction y
        '''        
        return (pred_y == true_y).sum() / len(true_y)

def Main():
    market_ticker = 'hs300'
    stock_ticker = '600859.SH'
    solver = GaussianNaiveBayesMethod()    
    market_data = solver.LoadData([market_ticker], 20190304, 20200409, 1, ['ts_code', 'trade_date','rate_of_increase'])
    stock_data = solver.LoadData([stock_ticker], 20190305, 20200410, 1, ['ts_code', 'trade_date','rate_of_increase'])
    X = market_data[0, :, 2]
    X = X.reshape((len(X),1))
    y = stock_data[0, :, 2]
    y = CalcPredictResult(y)
    X_train, y_train, X_test, y_test = UtilFuncs.SplitData(X, y, 2.0 / 3.0, True)
    solver.Fit(X_train,y_train)
    pred_y = solver.Predict(X_test)
    accuracy = solver.Accuracy(pred_y, y_test)
    print(accuracy)
    


    
    pass

Main()