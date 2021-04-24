import numpy as np
import scipy
class UtilFuncs:
    @staticmethod
    def R2(pred_Y :np.array, true_Y :np.array):
        '''
        R-squared
        '''
        r2 = 1.0 - ( (true_Y - pred_Y) ** 2 ).sum() / (( true_Y - true_Y.mean() ) ** 2).sum()
        return r2

    @staticmethod
    def CompareRandomVariable( y1 :np.array, y2 :np.array):
        # H0:y1 == y2
        # H1:y1 != y2
        difference :np.array = y1 - y2
        t = (np.mean(difference))/(difference.std(ddof=1)/np.sqrt(len(difference)))
        # 95% confidence 
        return t, np.abs(t) <= 1.96

    @staticmethod
    def IsSignificant( X:np.array):
        '''
        is random variable X significant
        '''
        zeros = np.zeros(len(X))
        return UtilFuncs.CompareRandomVariable(X, zeros)
    
    @staticmethod
    def SplitData(X :np.array, y :np.array, train_perc, is_shuffle):
        # build indices
        data_cnt :int = len(y)
        indices = np.arange(data_cnt)
        # shuffle
        if is_shuffle:
            np.random.shuffle(indices)
        # pick the two thirds as the training
        end_idx = int(data_cnt * train_perc)
        X_train = X[indices[:end_idx]]
        X_test = X[indices[end_idx:]]
        y_train = y[indices[:end_idx]]
        y_test = y[indices[end_idx:]]        
        return X_train,y_train, X_test, y_test

    @staticmethod
    def Normalize( X :np.array):
        return (X - X.min(0)) / X.ptp(0)


    @staticmethod
    def OneHotEncode( y :np.array):
        '''
        y : int array
        '''
        
    
        