from Data.UseData import read_data
from InvestmentAnalystSystem.FactorAnalysis.StyleFactorInfo import StyleFactorInfo
class FactorPortfolioBuilder:
    def __init__(self):
        pass

    
    def SingleFactorBuilderAtT(self, stock_array: np.array, sort_column: int, perc_portfolio :float):
        '''
        stock_array: (stock_num, feature_num)
        '''
        # we sort the stock list by the factor variable
        indices :np.array = stock_array[:, sort_column].argsort()
        total_num :int = stock_array.shape[0]
        pick_num :int = total_num * perc_portfolio
        low_indices = indices[:pick_num]
        high_indices = indices[-pick_num:]
        return low_indices, high_indices

    def BuildFactor(self, factor_columnname, start_date, end_date, rebalance_distance, yieldcheck_distance):
        # ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])
        yield_columnname = "rate_of_increase_" + str(yieldcheck_distance)
        yield_columnindex = 2
        factor_columnindex = 3
        need_columns =  ['ts_code', 'trade_date', yield_columnname, factor_columnname]
        reader = read_data()
        # all the stock data of the whole market
        succ, info, asset_data = reader.get_daily_data( None ,start_date, end_date, 1,
                   need_columns)
        if not succ:
            return False
        factor_list = []
        current_date = start_date
        # pick the portfolio
        for t in range(0, asset_data.shape[1], rebalance_distance):
            stocks_at_t = asset_data[:, t, :]
            low_indices, high_indices = self.SingleFactorBuilderAtT(stocks_at_t, factor_columnindex, 0.05)
            # now we need to build a long-short portfolio as a factor
            style_factor = StyleFactorInfo()
            style_factor.Init(current_date, high_indices, low_indices, asset_data, yield_columnindex)
            factor_list.append(style_factor)
            # move the date forward
            current_date += rebalance_distance
            

            

