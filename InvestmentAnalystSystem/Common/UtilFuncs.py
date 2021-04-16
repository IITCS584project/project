import numpy as np
class UtilFuncs:
    @staticmethod
    def R2(pred_Y :np.array, true_Y :np.array):
        r2 = 1.0 - ( (true_Y - pred_Y) ** 2 ).sum() / (( true_Y - true_Y.mean() ) ** 2).sum()
        return r2