import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
class UtilFuncs:
    @staticmethod
    def R2(pred_Y :np.array, true_Y :np.array):
        '''
        R-squared
        '''        
        r2 = r2_score(true_Y, pred_Y)
        return r2

    @staticmethod
    def AdjustedR2( feature_num :int, pred_Y :np.array, true_Y :np.array):
        sample_num = len(pred_Y)
        r2 = UtilFuncs.R2(pred_Y, true_Y)
        adjusted_r2 = 1 - (sample_num - 1) / ( sample_num - feature_num - 1) * (1 - r2)
        return adjusted_r2

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
        result = stats.ttest_1samp(X, 0.0)
        t = result.statistic
        p = result.pvalue
        # 1.65,1.96,2.58
        # 90%, 95%, 99%
        if t > 1.65:
            return True, t, p
        else:
            return False, t, p
       
    
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
        pass
    
    @staticmethod
    def TransformDailyYieldWithEpoch( daily_yield, start_date, due_date, date_column, yield_column, epoch ):
        '''
        ????????????????????????????????????
        epoch:????????????????????????, epoch >= 1

        ????????????????????????????????????????????????1???10???10?????????????????????epoch=2
        ?????????????????????2,4,6,8,10???????????????
        '''
        if epoch < 1:
            epoch = 1
        end_tindex = np.where(daily_yield[:, date_column] == due_date)[0][0]
        start_tindex = np.where(daily_yield[:, date_column] == start_date)[0][0]
        yield_list = []
        for t in range(end_tindex, start_tindex, -epoch):
            new_yield = 1.0
            for k in range(epoch):
                today_yield = daily_yield[t - k]
                today_yield = today_yield[yield_column]
                new_yield *= 1.0 + today_yield * 0.01
            yield_list.append(new_yield)
        yield_list.reverse()
        yield_list = np.array(yield_list)
        return (yield_list - 1.0) * 100


    
        