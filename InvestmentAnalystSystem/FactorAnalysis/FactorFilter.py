from InvestmentAnalystSystem.LinearAnalyst.MultiFactorSystem import MultiFactorSystem
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
        
        # first step, calculate the exposure of specific risk
        # there are two kinds of exposure. One of them is 




