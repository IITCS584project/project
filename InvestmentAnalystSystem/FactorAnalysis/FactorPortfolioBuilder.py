# -*- coding: UTF-8 -*-
from Data.UseData import read_data
from InvestmentAnalystSystem.FactorAnalysis.StyleFactorPortolioInfo import StyleFactorPortolioInfo
class FactorPortfolioBuilder:
    def __init__(self):
        pass

    
    def SingleFactorBuilderAtT(self, stock_array: np.array, sort_column: int, perc_portfolio :float):
        '''
        单变量排序，将所有股票按照某个变量排序，取分值最高和分值最低的两组股票，        
        stock_array: (stock_num, feature_num)
        '''
        # we sort the stock list by the factor variable
        indices :np.array = stock_array[:, sort_column].argsort()
        total_num :int = stock_array.shape[0]
        pick_num :int = total_num * perc_portfolio
        low_indices = indices[:pick_num]
        high_indices = indices[-pick_num:]
        return low_indices, high_indices

    def BuildFactor(self, factor_columnname, start_date, end_date, rebalance_distance):
        '''
        单变量排序后，取分值最高和分值最低的两组股票，
        构建多空对冲组合，近似作为因子收益率
        '''
        # ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])
        yield_columnname = "rate_of_increase_" + str(1)
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
        # 因子需要定期reblance，每次rebalance之后，因子内的股票都不一样
        for t in range(0, asset_data.shape[1], rebalance_distance):
            stocks_at_t = asset_data[:, t, :]
            low_indices, high_indices = self.SingleFactorBuilderAtT(stocks_at_t, factor_columnindex, 0.05)
            # now we need to build a long-short portfolio as a factor
            # 构造风格因子，它是一个多空组合
            style_factor = StyleFactorInfo()
            style_factor.Init(current_date, current_date + rebalance_distance - 1, t, high_indices, low_indices, asset_data, yield_columnindex)
            factor_list.append(style_factor)
            # move the date forward
            current_date += rebalance_distance
        
    
    def CalculateStyleFactorSignificant(self, begin_date, end_date, distance):
        '''
        计算风格因子在一段时期内的时间序列收益率是否显著不为0
        '''

        pass

            


