# -*- coding: UTF-8 -*-
class StyleFactorPortfolioInfo:
    '''
    style factor is a kind of factor which is realted with the stock itself
    such as E/P, B/P
    '''

    def __init__(self):
        pass
    def Init(self, begin_date, end_date, index, long_stockindices, short_stockindices, wholemarket_stocks, yieldcolumnindex):
        self.mBeginDate = begin_date
        self.mEndDate = end_date
        self.mIndex = index
        self.mLongIndices = long_stockindices
        self.mShortIndices = short_stockindices
        self.mMarketStocks = wholemarket_stocks
        self.mYieldColumnIndex = yieldcolumnindex



    def CalculateYieldEqualWeighted(self, start_date, end_date, distance, date_column, yield_column):
        # assuming it is equal weighted portfolio
        # 等权重portfolio
        if end_date > self.mEndDate:
            end_date = self.mEndDate
        # 多头股票
        long_stocks = self.mMarketStocks[self.mLongIndices, :]
        long_startindex = np.where(long_stocks[:, date_column, :] == start_date) 
        long_endindex = np.where(long_stocks[:, date_column, :] == end_date)
        long_num = long_stocks.shape[0]
        
        # 空头股票
        short_stocks = self.mMarketStocks[self.mShortIndices, :]
        short_startindex = np.where(short_stocks[:, date_column, :] == start_date)
        short_endindex = np.where(short_stocks[:, date_column, :] == end_date)
        short_num = short_stocks.shape[0]
        # 期初多头空头等金额
        netval_long = np.full()
        netval_short = 0.5
        
        epochs = 0
        for k in range(long_startindex, long_endindex + 1, distance):
            stocks_at_t = long_stocks[:, k, :]
            yeilds_at_t = stocks_at_t[:, yield_column]
            long_yield += yeilds_at_t.mean()
            epochs += 1
        
        long_yield /= epochs
        


        short_yield = 0.0
        epochs = 0
        for k in range(short_startindex, short_endindex + 1, distance):
            stocks_at_t = long_stocks[:, k, :]
            yeilds_at_t = stocks_at_t[:, yield_column]
            short_yield += yeilds_at_t.mean()
            epochs += 1
        short_yield /= epochs

        final_yield = long_yield - short_yield
        return final_yield

