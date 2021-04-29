from InvestmentAnalystSystem.LinearAnalyst.MultiFactorSystem import MultiFactorSystem
from InvestmentAnalystSystem.FactorAnalysis.FactorPortfolioBuilder import FactorPortfolioBuilder
class FactorAnalyzer:
    def __init__(self):
        pass

    def ComputeRiskExposure(self, X, y):
        # first, calculate the beta exposed to the factor
        solver = MultiFactorSystem()
        solver.Fit(X, y)
        param_map = solver.ShowParameters()
        beta = param_map['fc1.weight'][0,0]
        bias = param_map['fc1.bias'][0]
        return beta, bias

    def ComputeReturnTStatistics( self, highgroup_yields :np.array, lowgroup_yields :np.array):
        '''
        calculate the t statistics of yields of high group and low group
        '''
        times = len(highgroup_yields)
        rh_mean = highgroup_yields.mean()
        rl_mean = lowgroup_yields.mean()
        std_h = highgroup_yields.std(ddof = 1)
        std_l = lowgroup_yields.std(ddof = 1)
        std_hl = np.sqrt((std_h ** 2 + std_l ** 2) / len(times))
        t = (rh_mean - rl_mean) / std_hl

        # 95% confidence 
        return t, np.abs(t) > 1.96

    def IsSignificant(self, t):
        # 95% confidence
        return np.abs(t) > 1.96

    def LoadStockData(self, ticker_list, risk_yield, start_date, end_date):
        
        
        reader = read_data()
        succ, info, asset_data = reader.get_daily_data( ticker_list ,start_date, end_date, 1,
                    ['ts_code', 'trade_date', 'rate_of_increase_1', 'vol', 'rate_of_increase_7'])

        
        pass

    

    def TestSingleFactorValidity(self, stocks_at_t, factor_column):
        
        low_indices, high_indices = FactorPortfolioBuilder.SingleFactorBuilder(stock_list, factor_column, 0.1)

        pass