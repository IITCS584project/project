from InvestmentAnalystSystem.FactorAnalysis.StyleFactorPortfolioInfo
class StyleFactorInfo:
    def __init__(self):
        self.mPortfolios = []
        pass

    def AddPortfolio(self, portfolio :StyleFactorPortfolioInfo):
        self.mPortfolios.append(portfolio)
        pass

    def 