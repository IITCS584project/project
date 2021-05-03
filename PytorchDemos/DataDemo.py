from Data.UseData import read_data
def Main():
     #code=['sh','sz','hs300','sz50','zxb','cyb']

    obj_read=read_data()
    succ, info, market_info = obj_read.get_daily_data( ['hs300'], [] ,20200308, 20200315, 1,
                    ['ts_code', 'trade_date', 'vol', 'rate_of_increase_1' , 'rate_of_increase_3', 'rate_of_increase_7', 'rate_of_increase_20'])

    print(succ, info, market_info)

Main()