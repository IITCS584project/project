from InvestmentAnalystSystem.FactorAnalysis.SpreadPortfolioInfo import SpreadPortfolioInfo
import numpy as np
from scipy import stats
class StyleFactorInfo:
    def __init__(self):
        self.mPortfolios = []
        pass

    def Init( date_column, yield_column):
        self.mDateColumn = date_column
        self.mYieldColumn = yield_column

    def AddPortfolio(self, portfolio :SpreadPortfolioInfo):
        self.mPortfolios.append(portfolio)
        pass
    
    

    def CalculateTimeSeriesYield(self, start_date, end_date, distance):
        '''
        计算因子时间序列收益率
        
        start_date : 开始时间
        end_date : 结束时间
        distance : 
        '''
        # calculate the average return in the time series
        # 首先计算一个时间序列内的平均收益率
        portfolio : SpreadPortfolioInfo = None
        # 时间序列的收益率
        ts_yields = []
        for t in range(start_date, end_date + 1, distance):
            # 计算各段时间的收益率
            begin_date = t
            due_date = t + distance
            if due_date > end_date:
                due_date = end_date
            is_begin = False
            total_yield :float = 1.0
            # portfolio是按照时间顺序排列的
            for portfolio in self.mPortfolios:
                if not portfolio.IsValidTime(start_date) and not portfolio.IsValidTime(due_date):
                    continue
                portfolio_yield = portfolio.CalculateYieldEqualWeighted(begin_date, due_date, self.mDateColumn, self.mYieldColumn)
                if portfolio_yield == False:
                    continue
                total_yield *= portfolio_yield
            
            ts_yields.append(total_yield)
        ts_yields = np.array(ts_yields)
        return ts_yields

    def IsTimeSeriesYieldSignificant(self, ts_yield :np.array):
        '''
        当前时间序列的收益率是否显著不为0
        '''
        result = stats.ttest_1samp(ts_yield, 0.0)
        t = result.statistic
        p = result.pvalue
        # 1.65,1.96,2.58
        # 90%, 95%, 99%
        if t > 1.65:
            return True, t, p
        else:
            return False, t, p

    