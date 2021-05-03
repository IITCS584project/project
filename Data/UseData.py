 #!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import pandas as pd
import os
import numpy as np
import fnmatch
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
path=os.path.split(os.path.realpath(__file__))[0]
#print(path)

class read_data():

    def __init__(self):
        
        self.path=path+'/data_csv'
        
    def clac_tech_index(self,df):

        #增加EMA12列=前一日EMA（12）×11/13＋今日收盘价×2/13
        #增加EMA26列= 前一日EMA（26）×25/27＋今日收盘价×2/27
        #今日DIF=EMA12-EMA26
        #今日DEA value= 昨天 MACD*0.8+今日DIF*0.2
        #MACD=2*（今日DIF-今日DEA）
        #df = df.sort_values(["trade_date"], ascending=False)

        #df['EMA12']=df['close']
        #df['EMA26']=df['close']
        #df['DIF']=df['EMA12']-df['EMA26']
        df.loc[0,'EMA12']=df.loc[0,'close']
        df.loc[0,'EMA26']=df.loc[0,'close']
        df.loc[0,'DEA']=0
        for i in range(1,len(df)):

            df.loc[i,'EMA12']=round(df.loc[i,'close']*2/13+df.loc[i-1,'EMA12']*11/13,2)
            df.loc[i,'EMA26']=round(df.loc[i,'close']*2/27+df.loc[i-1,'EMA26']*25/27,2)
        df['DIF']=df['EMA12']-df['EMA26']
        
        for i in range(1,len(df)):

            df.loc[i,'DEA'] =round(df.loc[i-1,'DEA']*8/10+df.loc[i,'DIF']*2/10,2)
        df['MACD']  =round((df['DIF']- df['DEA'])*2,2)

        df['MA7']=df['close']
        df['MA15']=df['close']
        df['MA30']=df['close']
        for i in range(6,len(df)):
            for j in range(0,6):
                df.loc[i,'MA7']+=df.loc[i-j,'close']
            df.loc[i,'MA7']=df.loc[i,'MA7']/7
        for i in range(14,len(df)):
            for j in range(0,14):
                df.loc[i,'MA15']+=df.loc[i-j,'close']
            df.loc[i,'MA15']=df.loc[i,'MA15']/15
        for i in range(29,len(df)):
            for j in range(0,29):
                df.loc[i,'MA30']+=df.loc[i-j,'close']
            df.loc[i,'MA30']=df.loc[i,'MA30']/30
        return df

    def get_daily_data(self,codelist,except_codelist='',start_date='',end_date='',distance=1,columns=''):
        
        # 读标的物列表
        # param  code  必填，标的物编码  e.g. 000001.SZ
        # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部
        # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部
        __code_list=[]
        #如果传入的code不是list ,直接异常
        if type(codelist) != type([]):
            return (False,'codelist is need list type, your  parameter is '+str(type(codelist)),None)

        stock_list=self.get_stock_list()
        #如果传入的code 有不在范围内的，直接异常
        for code in codelist:
            if code not in stock_list['ts_code'].tolist():
                return (False,'code '+str(code)+' not in datalist',None)
            else:
                __code_list.append(code)
        #如果不传入指定code，使用全部
        if len(codelist)==0:
            __code_list=stock_list['ts_code'].tolist()

        #这个时候已经完整生成了__code_list了, 再删除except的内容

        if except_codelist!='':
            if type(except_codelist) != type([]):
                return (False,'except_codelist is need list type, your  parameter is '+str(type(codelist)),None)
            for r in except_codelist:
                if r in __code_list:
                    __code_list.remove(r)

        if len(__code_list)==0:
            return (False,'you request a none array','')  

        res_list=[]
        isopen_drop_list={}

        #先取出来所有交易日历

        df_trade_cal=self.read_trade_cal(start_date=start_date,end_date=end_date)
        df_trade_cal['trade_date']=df_trade_cal['cal_date']
        df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
        for code in __code_list:
            
            df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
            n=[]#需要收益率的list
            m=[]#需要计算当日到未来收益率的list
            if columns!='':
                for str_ in columns:
                    if fnmatch.fnmatch(str_, 'rate_of_increase_*'):
                        n.append(int(str_.split('_')[-1]))
                    if fnmatch.fnmatch(str_, 'rate_of_increase_next_*'):
                        m.append(int(str_.split('_')[-1]))
            #此时n应该为一个list  ,ex：[4,5,6]要计算的n日累计涨幅
            for x in n:
                df['rate_of_increase_' + str(x)] = round(((df['close'] / df.shift(periods=-x)['close']) - 1) * 100,3)
            for x in m:
                df['rate_of_increase_next_' + str(x)] = round((( df.shift(periods=x)['close'] / df['close']) - 1) * 100,3)
            #为df增加每个n日涨幅列

            df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
            df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df # 截取日期片
            df=df.sort_values(["trade_date"],ascending=True) #重新排序 时间从上到下，变为从早到晚

            ####### 202104030 增加跟交易日匹配，返回每只股票缺失的交易日信息
            df=pd.merge(df,df_trade_cal,how='right',on=['trade_date'],sort=False,copy=True,suffixes=('','_y'))
            df.drop(columns=['Unnamed: 0_y'],inplace=True)
            if len(df[df['ts_code'].isna()]['trade_date'].to_list())>0:
                isopen_drop_list[code]=df[df['ts_code'].isna()]['trade_date'].to_list()

            ####### 返回信息后，去除了交易日历里带空的内容
            df=df[df['ts_code'].notnull()]
            #######
            df=df.reset_index(drop=True) # 重新定义索引
            columns_clac=[]
            columns_clac.append('EMA12')
            columns_clac.append('EMA26')
            columns_clac.append('DIF')
            columns_clac.append('DEA')
            columns_clac.append('MA7')
            columns_clac.append('MA15')
            columns_clac.append('MA30')
            columns_clac.append('MACD')

            if len(list(set(columns_clac).intersection(set(columns))))>0: #有需要计算指标的时候，再计算计算指标
                df = self.clac_tech_index(df) # 增加计算指标

            for i in df.index:
                if i%distance!=0: #用index 取余 distance，余0返回，其他去掉
                    df.drop(index=i,inplace=True)



            df.drop(columns=['Unnamed: 0'],inplace=True) # 文件内带有默认数字列，特定为了删除这个

            df=df if  columns=='' else df[columns]
            res=df.values
            res_list.append(res)

        #print(df)

        #20210430 如果有因为交易日历剔除数据就返回false
        if len(isopen_drop_list)>0:
            return (False,isopen_drop_list,np.array(res_list))
        
        #返回的时候判断特定值是不是有none，有就带上false
        res= (True,'',np.array(res_list)) if sum(df.isna().sum().to_list())==0 else  (False,'contains nan',np.array(res_list))
        return res

    def get_stock_list(self):

        df = pd.read_csv(self.path + "/stocklist.csv", sep=",")
        return df

    def get_daily_data_only_stock(self,codelist,except_codelist='',start_date='',end_date='',distance=1,columns=''):
        __code_list=[]
        #如果传入的code不是list ,直接异常
        if type(codelist) != type([]):
            return (False,'code is need list type, your  parameter is '+str(type(codelist)),None)
        stock_list=self.get_stock_list()
        #剔除掉指数数据
        stock_list=stock_list[stock_list['industry']!='后补数据']
        
        for code in codelist:
            if code not in stock_list['ts_code'].tolist():
                return (False,'code '+str(code)+' not in datalist',None)
            else:
                __code_list.append(code)
        #如果不传入指定code，使用全部
        if len(codelist)==0:
            __code_list=stock_list['ts_code'].tolist()

        return self.get_daily_data(__code_list,except_codelist=except_codelist,start_date=start_date,end_date=end_date,distance=distance,columns=columns)
    def get_daily_data_clac(self,code,start_date='',end_date=''):
        #tips：因为有的指标是需要昨日数据的，所以如果是该股票开盘第一天的话，一些rate 会是 NaN
        #close_rate_of_increase=今日收盘价/昨天收盘价
        #avg_rate_of_increase=今日四项平均值/昨日四项平均值
        #open_today_rate_of_increase=今日收盘价／今日开盘价
        #middle_rate_of_increase=「1/2(今日最高+今日最低)」/「1/2(昨日最高+昨日最低)」
        #分类参数 档位{key1:value1 ,key2:value1~value2 ,key3 value3~value4,key4value4~}

        code=str(code.upper())
        df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
        df=df.sort_values(["trade_date"],ascending=True)
        df['close_rate_of_increase']=(df['close']/df['pre_close']-1)*100
        df['avg_for_OHLC']=(df['close']+df['open']+df['high']+df['low'])/4
        df['avg_for_HL']=(df['high']+df['low'])/2
        df['open_today_rate_of_increase']=(df['open']/df['close']-1)*100
        df['avg_rate_of_increase']=((df['avg_for_OHLC']/df.shift(periods=1)['avg_for_OHLC'])-1)*100
        df['middle_rate_of_increase']=((df['avg_for_HL']/df.shift(periods=1)['avg_for_HL'])-1)*100
        del df['avg_for_OHLC']
        del df['avg_for_HL']
        df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df
        return df

    def save_local(self,name,msg):
        path=self.path+'/data_local'
        if not os.path.exists(path):
            os.mkdir(path) 
        if type(msg) != type(np.array([0])):
            return ('msg is not type np.array,your  parameter is '+str(type(msg)))


        np.save(path+'/'+name+'.npy',msg,allow_pickle=True)
    def read_local(self,name):
        path=self.path+'/data_local'
        if not os.path.exists(path):
            os.mkdir(path) 
        try:        
            res=np.load(path+'/'+name+'.npy',allow_pickle=True)
        except Exception as e:
            print(str(e))
            return 
        return res

    def read_trade_cal(self,start_date,end_date):
        df = pd.read_csv(self.path + "/trade_cal.csv", sep=",")
        df = df.loc[df["cal_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["cal_date"] <= end_date] if end_date != '' else df # 截取日期片
        return df
    #20210502 增加返回限定日期内有交易股票的范围
    def get_trade_cal_stock_list(self,start_date,end_date):
        df_trade_cal=self.read_trade_cal(start_date=start_date,end_date=end_date)
        df_trade_cal=df_trade_cal[df_trade_cal['is_open']==1]
        trade_date_=df_trade_cal['cal_date'].tolist()
        trade_date_min,trade_date_max=trade_date_[0],trade_date_[-1]



        stock_list=self.get_stock_list()
        stock_list=stock_list[stock_list['industry']!='后补数据']
        code__=[]
        for code in stock_list['ts_code'].tolist():
            
            df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
            if len(df[df['trade_date']==trade_date_min])==1 and len(df[df['trade_date']==trade_date_max])==1:
                code__.append(code)

        return code__


if __name__ == '__main__':

    #code=['sh','sz','hs300','sz50','zxb','cyb']

    obj_read=read_data()
    succ, info, market_info = obj_read.get_daily_data( ['hs300'], [] ,20200308, 20200315, 1,
                    ['ts_code', 'trade_date', 'vol', 'rate_of_increase_1' , 'rate_of_increase_3', 'rate_of_increase_7', 'rate_of_increase_20'])
    print(market_info)
    obj_read.save_local('test',market_info)

    print(333)
    print(obj_read.read_local('test'))
    
#3D 混合
#交易日剔除计算 ok
#剔除其他 ok
# 只有用到计算指标的时候再计算 ok


#ts_code		股票代码
#trade_date		交易日期
#open		开盘价
#high		最高价
#low		最低价
#close		收盘价
#vol		成交量 （手）
#amount		成交额 （千元）
#rate_of_increase_*(*只可以是数字) 过去n日累计涨幅比例
#rate_of_increase_next_*(*只可以是数字) 未来n日累计涨幅比例
#turnover_rate		换手率（%）
#turnover_rate_f		换手率（自由流通股）
#volume_ratio		量比
#pe		市盈率（总市值/净利润， 亏损的PE为空）
#pe_ttm		市盈率（TTM，亏损的PE为空）
#pb	float	市净率（总市值/净资产）
#ps	float	市销率
#ps_ttm		市销率（TTM）
#dv_ratio		股息率 （%）
#dv_ttm		股息率（TTM）（%）
#total_share		总股本 （万股）
#float_share		流通股本 （万股）
#free_share		自由流通股本 （万）
#total_mv		总市值 （万元）
#circ_mv		流通市值（万元）
#EMA26
#EMA12
#MA7
#MA5
#MA15
#DIF
#DEA
#MACD

