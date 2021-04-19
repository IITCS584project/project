from Data.UseData import read_data
import numpy as np

class StockDataProvider:
    @staticmethod
    def GetStockDataForPredict( asset_ticker, market_ticker, start_date, end_date ):
        reader = read_data()
        asset_data = reader.get_daily_data( [asset_ticker, market_ticker] ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'rate_of_increase', 'vol'])        
        # remove 1st row, the 1st yield is na
        yield_data = asset_data[0, 1:, 2]        
        # remove the last row,I use the chracteristics of yesterday to predict today's price
        yesterday_data = asset_data[0, :-1,3:]
        # use the market data of yesterday to predict today's price
        market_data = asset_data[1, :-1, 2]

        X = yesterday_data
        y = yield_data
        return X, y
    