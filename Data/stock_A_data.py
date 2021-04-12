#!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
import datetime
import sys
import csv
import json
import tushare as ts
import pandas as pd
import requests
import os
from interval import Interval
ts.set_token('6b57e4c863aa1e2d55c8ce62742130dd38332aafe1c43f2666dedfe5')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
pro = ts.pro_api()
path=os.path.split(os.path.realpath(__file__))[0]
#print(path)
class get_data_from_tushare():

    def __init__(self):
        self.pro = ts.pro_api()
        self.data=''
        self.path=path+'/data_csv'

    def write_detail_data_signle_ts(self,code):
        df=ts.get_hist_data(code)
        df['trade_date']=df.index
        for i in range(0,df.shape[0]):
            df.loc[df.iloc[i]['trade_date'],['trade_date']]=df.iloc[i]['trade_date'].replace("-",'')
        df['ts_code']=code
        df['pre_close']=0
        df['change']=0
        df['pct_chg']=0
        df['vol']=df['volume']
        df['amount']=0
        df=df.reset_index()
        df=df[['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount']]
        df.to_csv(self.path+'/' + code + '.csv')
        
    def get_stock_list(self):
        #获取标的物的列表
        self.data = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        self.data.to_csv(self.path+'/stocklist.csv')
    def write_detail_data(self):
        #从tushare按照标的物列表逐个写入日数据
        for i in range(0,self.data.shape[0]):
            ts_code = self.data.iloc[i]['ts_code']
            df = pro.daily(ts_code=ts_code)
            df.to_csv(self.path+'/' + ts_code + '.csv')
            time.sleep(1) # 对方接口要求控制频次，每分钟不能超过500次，文件有大有小，是有概率超的，所以统一sleep下
            del (ts_code)
    def write_detail_data_minutes(self,pagesize):
        #从sina按照标的物列表逐个写入5分钟数据
        #对方有爬虫限制，目前暂时看每次减缓10s实验中
        for i in range(0,self.data.shape[0]):
            ts_code = self.data.iloc[i]['ts_code']
            ts_code=ts_code.split('.')[1]+ts_code.split('.')[0]
            request_url='http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol='+ts_code+'&scale=5&ma=no&datalen='+str(pagesize)
            response = requests.get(request_url)
            print(response.content)
            res = json.loads(response.content)

            time.sleep(10)
            df = pd.DataFrame(res)
            df.to_csv(self.path+'/minutes/' + ts_code + 'minute.csv')
            del(df)

class read_data():

    def __init__(self):
        self.path=path+'/data_csv'


    def get_daily_data(self,code,start_date='',end_date=''):
        
        # 读标的物列表
        # param  code  必填，标的物编码  e.g. 000001.SZ
        # param  code  不必填，开始时间，没填写的时候，直到取最早，开始结束都没填写的时候取全部
        # param  code  不必填，结束时间，没填写的时候，直到取最晚，开始结束都没填写的时候取全部
        code=str(code.upper())
        df=pd.read_csv(self.path+"/"+code+".csv", sep=",")
        df = df.loc[df["trade_date"] >= start_date] if start_date != '' else df
        df = df.loc[df["trade_date"] <= end_date] if end_date != '' else df
        df=df.sort_values(["trade_date"],ascending=True)
        return df
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
        return df
    def set_learn_tag(self,df,taglists,columns):
        #taglist dict 
        #column str 枚举
        #仅选择一个指标作为打tag打指标，根据这个指标内容增加一个learn_tag列，根据taglists的key进行打标签

        _columns=['close_rate_of_increase','open_today_rate_of_increase','avg_rate_of_increase','middle_rate_of_increase']
        if columns not in _columns:
            print('columns param need to one str of ['+','.join(_columns)+']')
        df['learn_tag']=None
        for i in df.index:
            for key in taglists.keys():
                index_line=df.index.get_loc(i)
                if df.iloc[index_line][columns] in taglists.get(key):
                        df.loc[i,['learn_tag']]=key 
        return df


if __name__ == '__main__':
    code=['sh','sz','hs300','sz50','zxb','cyb']
    for r in code:
        obj_=get_data_from_tushare()
        obj_.write_detail_data_signle_ts(r)
        # obj_read=read_data() #声明对象
    # df=obj_read.get_daily_data_clac('000001.SZ',start_date=20200128,end_date=20210302)# 按照开始时间和结束时间取 000001.SZ这个标的物

    # taglist={}
    # #定义一个tag dict 增加自定义的分类标记,key的名称将会直接打在数据上
    # taglist['big_short']=Interval(float('-inf'),-3)# 将跌幅无穷到-3% 定义为大跌
    # taglist['small_short']=Interval(-3,-1)#将-3% 到-1% 定义为小跌
    # taglist['middle']=Interval(-1,1)
    # taglist['small_long']=Interval(1,3)
    # taglist['big_long']=Interval(3,float('inf'))
 
    # learn_tag_column='middle_rate_of_increase' # 选择根据要打标记的指标
    # #close_rate_of_increase=今日收盘价/昨天收盘价
    # #avg_rate_of_increase=今日四项平均值/昨日四项平均值
    # #open_today_rate_of_increase=今日收盘价／今日开盘价
    # #middle_rate_of_increase=「1/2(今日最高+今日最低)」/「1/2(昨日最高+昨日最低)」

    # res=obj_read.set_learn_tag(df,taglist,learn_tag_column) # 按middle_rate_of_increase 这个指标打标记，生成learn_tag标签，用于后续学习和分类
    # print(res)