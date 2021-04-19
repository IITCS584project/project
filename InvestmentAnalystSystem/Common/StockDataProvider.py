from Data.UseData import read_data
import numpy as np
import torch
from InvestmentAnalystSystem.Common.UtilFuncs import UtilFuncs

class StockDataProvider:
    @staticmethod
    def GetStockDataForPredict( asset_ticker, market_ticker, start_date, end_date ):
        reader = read_data()
        succ, info, asset_data = reader.get_daily_data( [asset_ticker, market_ticker] ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol'])        
        # remove the 1st row
        yield_data = asset_data[0,1:, 2]        
        # remove the last row,I use the chracteristics of yesterday to predict today's price
        yesterday_data = asset_data[0, :-1,3:]
        yesterday_data[0] = UtilFuncs.Normalize(yesterday_data[0])
        # use the market data of yesterday to predict today's price
        market_data = asset_data[1, :-1, 2]
        market_data = market_data.reshape((len(market_data),1))
        X = np.concatenate([yesterday_data, market_data], axis=1)
        y = yield_data
        return X, y
    

    @staticmethod
    def NpArrayToTensor(X):
        return torch.from_numpy(np.array(X, dtype=float)).float()
        