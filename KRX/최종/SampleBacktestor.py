import os
import numpy as np
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import unittest

plt.rcParams['figure.figsize'] = 17,9
rolling_date = 15

with open('/Users/donghanko/Desktop/database.pickle', 'rb') as f:
    database = pickle.load(f)

Stocklst = list(database.keys())
rep_data =  database[Stocklst[0]]

signal_dataframe = pd.DataFrame(index = rep_data.index[:-rolling_date+1])




ret = []
for idx in range(len(rep_data.index)-rolling_date):
    ret_value = (rep_data['Close'][idx+rolling_date] - rep_data['Close'][idx])/rep_data['Close'][idx]
    ret.append(ret_value+1)


def calculate_mdd(df):
    mdd_lst = []
    close_values = deque()
    for idx in range(len(df)):
        close_values.append(df['Close'][idx])
        if len(close_values) > rolling_date:
            close_values.popleft()
            max_val_in_box = max(close_values)
            latest_val_in_box = close_values[-1]
            mdd_in_box = (latest_val_in_box-max_val_in_box)/max_val_in_box
            mdd_lst.append(mdd_in_box)
    return mdd_lst


def calculate_ret_to_mdd(df):
    mdd_lst = []
    close_values = deque()
    for idx in range(len(df)):
        close_values.append(df['Close'][idx])
        if len(close_values) > rolling_date:
            close_values.popleft()
            max_val_in_box = max(close_values)
            latest_val_in_box = close_values[-1]
            mdd_in_box = (latest_val_in_box - max_val_in_box) / max_val_in_box
            ret_in_box = (close_values[-1]-close_values[0])/close_values[0]

            mdd_lst.append(ret_in_box/-1*mdd_in_box)
    return mdd_lst

column_names = ['Column' + str(i) for i in range(0, 479)]
merged_data = pd.DataFrame(index = database.keys(),columns=column_names)

def rank_function(df):
    df = df.sort_values(ascending=False)
    return {'long': df.head(200).index , 'short':df.tail(200).index}

if __name__=='__main__':
    for keys in list(database.keys()):
        merged_data.loc[keys] = calculate_ret_to_mdd(database[keys])

pnl_table = pd.DataFrame(columns=column_names)

def calculate_return(code_lst,ind):
    ret_lst = []
    for code in code_lst:
        target_table = database[code]['Close']
        ret = (target_table[ind+rolling_date]-target_table[ind])/target_table[ind]
        ret_lst.append(ret)
    return sum(ret_lst)/len(ret_lst)



if __name__=='__main__':
    i = 0
    portfolio_pnl = []
    while i<=len(database[Stocklst[0]].index)-rolling_date-1:

        dict = rank_function(merged_data[f'Column{i}'])
        for keys in dict.keys():
            if keys == 'long':
                long_ret = calculate_return(dict[keys],i) +1
            else:
                short_ret = -1*(calculate_return(dict[keys],i)) +1

        print(long_ret,short_ret)
        print((long_ret + short_ret)/2)
        portfolio_pnl.append((long_ret+short_ret)/2)
        i+=1



plt.plot(database[Stocklst[0]].index[rolling_date:],portfolio_pnl)
print(sum(portfolio_pnl)/len(portfolio_pnl))
plt.show()















