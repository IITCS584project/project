import numpy as np
class PredictResultType:
    Flat = 0,
    SmallRise = 1,
    BigRise = 2,
    SmallDrop = 3,
    BigDrop = 4

def CalcPredictResult(asset_yield :np.array):
    result = np.full(asset_yield.shape, PredictResultType.BigRise)    
    result[asset_yield < -5] = PredictResultType.BigDrop
    result[ (-5 <= asset_yield) & (asset_yield < -1)] = PredictResultType.SmallDrop
    result[ (-1 <= asset_yield) & (asset_yield < 1)] = PredictResultType.Flat
    result[(1 <= asset_yield) & (asset_yield < 5)] = PredictResultType.SmallRise
    return result

class RiseDropPredictResultType:
    Rise = 0,
    Drop = 1

def CalcRiseDropPredictResult(asset_yield :np.array):
    result = np.full(asset_yield.shape, RiseDropPredictResultType.Rise)
    result[asset_yield < 0] = RiseDropPredictResultType.Drop
    return result
