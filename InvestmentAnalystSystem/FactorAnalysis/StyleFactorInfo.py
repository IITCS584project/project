class StyleFactorInfo:
    def __init__(self):
        pass
    def Init(self, begin_date, end_date, long_stockindices, short_stockindices, wholemarket_stocks, yieldcolumnindex):
        self.mBeginDate = begin_date
        self.mEndData = end_date
        self.mLongIndices = long_stockindices
        self.mShortIndices = short_stockindices
        self.mMarketStocks = wholemarket_stocks
        self.mYieldColumnIndex = yieldcolumnindex

    def CalculateYieldEqualWeighted(self, start_date, end_date, distance, date_column, yield_column):
        # assuming it is equal weighted portfolio
        long_stocks = self.mMarketStocks[self.mLongIndices, :]
        long_startindex = np.where(long_stocks[:, date_column, :] == start_date) 
        long_endindex = np.where(long_stocks[:, date_column, :] == end_date)
        long_yield = 0.0
        
        epochs = 0
        for k in range(long_startindex, long_endindex + 1, distance):
            stocks_at_t = long_stocks[:, k, :]
            yeilds_at_t = stocks_at_t[:, yield_column]
            long_yield += yeilds_at_t.mean()
            epochs += 1
        
        long_yield /= epochs

    
        short_stocks = self.mMarketStocks[self.mShortIndices, :]
        short_startindex = np.where(short_stocks[:, date_column, :] == start_date)
        short_endindex = np.where(short_stocks[:, date_column, :] == end_date)
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

