from Data.UseData import read_data
import numpy as np
import torch
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs
from InvestmentAnalystSystem.Common.PredictResult import PredictResultType,CalcRiseDropPredictResult

class StockDataProvider:
    @staticmethod
    def GetStockDataForPredict( asset_ticker, market_ticker, start_date, end_date ):
        reader = read_data()
        succ, info, asset_info = reader.get_daily_data( [asset_ticker], [] ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'rate_of_increase_next_5', 'vol', 'rate_of_increase_1', 'rate_of_increase_3', 
                    'rate_of_increase_7', 'rate_of_increase_10', 'pe', 'pb', 'ps',  
                    'turnover_rate', 'volume_ratio'])
        '''
        if not succ:
            print("Asset error", asset_ticker, info)
            error_data = asset_info[0, :, 2:]
            error_data = error_data.astype(np.float)
            is_nan = np.isnan(error_data)
            nan_pos = np.where(is_nan)
            return
        '''

        succ, info, market_info = reader.get_daily_data( [market_ticker], [] ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'vol', 'rate_of_increase_1' , 'rate_of_increase_3', 'rate_of_increase_7', 'rate_of_increase_20'])
        '''
        if not succ:
            print("Market error", info)
            return
        '''
        # remove the 1st row
        # 要与预测的模型
        # rate of increase next 3
        true_y = asset_info[0,:, 2]
        # remove the last row,I use the chracteristics of yesterday to predict today's price
        stock_chracteristics = asset_info[0, :,3:]
        # normalize vol
        stock_chracteristics[:, 0] = UtilFuncs.Normalize(stock_chracteristics[:,0])
        # use the market data of yesterday to predict today's price
        market_data = market_info[0, :, 2:]
        market_data[:, 0] = UtilFuncs.Normalize(market_data[:,0])
        #market_data = market_data.reshape((len(market_data),3))
        X = np.concatenate([stock_chracteristics, market_data], axis=1)
        y = true_y.reshape((len(true_y), 1))
        return X, y

    @staticmethod
    def GenStockData(stock_ticker, market_ticker, train_begindate, train_enddate, test_begindate, test_enddate):
        X_train, y_train = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, train_begindate, train_enddate)
        y_train = y_train.reshape(len(y_train))
        
        X_test, y_test = StockDataProvider.GetStockDataForPredict(stock_ticker, market_ticker, test_begindate, test_enddate)
        y_test = y_test.reshape(len(y_test))
        
        return X_train, y_train, X_test, y_test

    def GetStockDataForMomentum(asset_ticker, start_date, end_date):
        reader = read_data()
        succ, info, asset_data = reader.get_daily_data( [asset_ticker, market_ticker] ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])
    

    @staticmethod
    def NpArrayToTensor(X):
        return torch.from_numpy(np.array(X, dtype=float)).float()
    
    @staticmethod
    def DummyGenerateStockData(sample_num):
        X = np.array(range(10)) + 0.0
        y = 2 * X + 5.0
        #X += np.random.rand() - 0.5      
        X = X.reshape((len(X),1))  
        y = y.reshape((len(y),1))
        return X, y