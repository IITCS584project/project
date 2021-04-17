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
        #df['dea_per']=df.shift(periods=1)['DEA']
        df['MACD'] = (df['DIF']- df['DEA'])*2
        #sum(df.shift(periods=i)['close'])
        

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

    def get_daily_data(self,codelist,start_date='',end_date='',distance=1,columns=''):
        
        # 读标的物列表
        # param  code  必填，标的物编码  e.g. 000001.SZ
        # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部
        # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部
        __code_list=[]
        #如果传入的code不是list ,直接异常
        if type(codelist) == []:
            return 'code is need list type, your  parameter is '+str(type(codelist))
        stock_list=self.get_stock_list()
        #如果传入的code 有不在范围内的，直接异常
        for code in codelist:
            if code not in stock_list['ts_code'].tolist():
                return 'code '+str(code)+' not in datalist'
            else:
                __code_list.append(code)
        #如果不传入指定code，使用全部
        if len(codelist)==0:
            __code_list=stock_list['ts_code'].tolist()

        res_list=[]

        for code in __code_list:

            df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
            
            df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
            df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df # 截取日期片
            df=df.sort_values(["trade_date"],ascending=True) #重新排序
            df=df.reset_index() # 重新定义索引

            for i in df.index:
                if i%distance!=0: #用index 取余 distance，余0返回，其他去掉
                    df.drop(index=i,inplace=True)

            df['rate_of_increase']=((df['close']/df.shift(periods=1)['close'])-1)*100
            df=df[['ts_code','trade_date','open','high','low','close','change','vol','amount','rate_of_increase']]
            df=self.clac_tech_index(df)
            res=df.values if columns=='' else df[columns].values
            res_list.append(res)

        return  np.array(res_list)

    def get_stock_list(self):

        df = pd.read_csv(self.path + "/stocklist.csv", sep=",")
        return df


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




if __name__ == '__main__':

    #code=['sh','sz','hs300','sz50','zxb','cyb']

    obj_read=read_data() #声明对象
    #result=obj_read.get_daily_data('sh',start_date=20210405,end_date=20210412,distance=2,columns=['ts_code','trade_date','open','high','low','close','change','vol','amount','rate_of_increase'])
    result=obj_read.get_daily_data(codelist=['sh'],start_date=20200101,end_date=20210412,distance=1)
    
    # code 传入一个list, list内如果有data set 以外的code 返回错误 | 如果传入不是 list格式 ，返回错误  | 如果传入空list 返回所有已有数据股票的内容
    # 按照开始时间和结束时间取 000001.SZ这个标的物,distance=n n代表返回日期间隔
    # 返回第1天一定是>=start_date的第一个交易日的日期数据
    # 按照distance的间隔，返回所有交易日内的间隔日期数据
    
    #2021-04-17
    #Index(['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'change','vol', 'amount', 'rate_of_increase', 'EMA12', 'EMA26', 'DEA', 'DIF','MACD', 'MA7', 'MA15', 'MA30'],



