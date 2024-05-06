import pandas as pd
import FinanceDataReader as fdr
import numpy as np
from pykrx import stock

lst = pd.read_csv(r'C:\Users\John\OneDrive\바탕 화면\퀀트관련\KRX대회\최종\final_submission.csv',index_col=0)

lst = lst.sort_values(by='순위')

long_lst = lst.index[:200]
short_lst = lst.index[-200:]

if __name__== '__main_':
    long_lst_name = []
    short_lst_name = []
    for idx in long_lst:
        long_lst_name.append(stock.get_market_ticker_name(idx[1:]))
    for idx in short_lst:
        short_lst_name.append(stock.get_market_ticker_name(idx[1:]))

    print(long_lst_name)
    print(short_lst_name)

if __name__=='__main__':
    short_ret = 0
    long_ret = 0

    for idx in long_lst:
        long_ret += fdr.DataReader(idx[1:],start='2023-7-31')['Change'].values
        print('Long: ',long_ret)

    for idx in short_lst:
        short_ret -= fdr.DataReader(idx[1:], start='2023-7-31')['Change'].values
        print('Short: ',short_ret)

    print('Long Total: ', long_ret)
    print('Short Total: ', short_ret)

    daily_lst = np.array((long_ret + short_ret)/400)


    print(daily_lst)
